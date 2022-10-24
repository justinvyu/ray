import logging
import time
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
)

import numpy as np
import pandas as pd
from pyarrow import parquet as pq
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsforecast import StatsForecast
from statsforecast.models import ETS, AutoARIMA

import ray
from ray import air, tune
from ray.air import Checkpoint, session
from ray.air.config import RunConfig, ScalingConfig
from ray.train.constants import TRAIN_DATASET_KEY
from ray.train.trainer import BaseTrainer, GenDataset

if TYPE_CHECKING:
    from ray.data.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


# __statsforecast_trainer_start__
class StatsforecastTrainer(BaseTrainer):
    def __init__(
        self,
        *,
        model_cls=None,
        datasets: Dict[str, GenDataset] = None,
        metrics: Optional[Dict[str, Any]] = None,
        freq: str = "D",
        parallelize_cv: bool = True,
        n_parallel_cv_jobs: int = -1,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        cv_params: Dict[str, Any] = None,
        statsforecast_params: Optional[Dict[str, Any]] = None,
        scaling_config: Optional[ScalingConfig] = None,
        run_config: Optional[RunConfig] = None,
        preprocessor: Optional["Preprocessor"] = None,
        **model_params,
    ):
        self.model_cls = model_cls
        self.model_params = model_params

        self.freq = freq
        self.metrics = metrics or {
            "mse": mean_squared_error,
            "mae": mean_absolute_error,
        }

        # Cross-validation parameters
        self.n_splits = n_splits
        self.test_size = test_size
        self.parallelize_cv = parallelize_cv
        self.n_parallel_cv_jobs = n_parallel_cv_jobs

        # Extra params to pass into `StatsForecast.cross_validation`
        self.cv_params = cv_params or {}

        # Extra params to pass when initializing `StatsForecast` object
        self.statsforecast_params = statsforecast_params or {}

        super().__init__(
            scaling_config=scaling_config,
            run_config=run_config,
            datasets=datasets,
            preprocessor=preprocessor,
            resume_from_checkpoint=None,
        )

    def _get_datasets(self) -> Dict[str, pd.DataFrame]:
        pd_datasets = {}
        for key, ray_dataset in self.datasets.items():
            pd_dataset = ray_dataset.to_pandas(limit=float("inf"))

            # Make sure statsforecast dataset columns are present
            assert "y" in pd_dataset.columns
            assert "unique_id" in pd_dataset.columns
            assert "ds" in pd_dataset.columns

            pd_datasets[key] = pd_dataset
        return pd_datasets

    def training_loop(self) -> None:
        # Construct the model from the class/params passed in
        self.model = self.model_cls(**self.model_params)

        # Get training dataset
        datasets = self._get_datasets()
        Y_train_df = datasets.pop(TRAIN_DATASET_KEY)

        if self.parallelize_cv:
            # Connect to the existing ray cluster via `auto`
            parallelism_kwargs = {
                "ray_address": "auto",
                "n_jobs": self.n_parallel_cv_jobs,
            }
        else:
            parallelism_kwargs = {"n_jobs": 1}

        # Initialize statsforecast with the model, using Ray as its parallel backend
        statsforecast = StatsForecast(
            df=Y_train_df,
            models=[self.model],
            freq=self.freq,
            **parallelism_kwargs,
            **self.statsforecast_params,
        )

        # Ex: with default test_size
        # Dataset with 12 items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        # n_splits = 5
        # test_size = 12 // (5 + 1) = 2
        # Splits:
        # Train = [0, 1]   Test = [2, 3]
        # Train = [0, 1, 2, 3]   Test = [4, 5]
        # ...
        self.test_size = self.test_size or len(Y_train_df) // (self.n_splits + 1)

        start_time = time.time()
        forecasts_cv = statsforecast.cross_validation(
            h=self.test_size,
            n_windows=self.n_splits,
            step_size=self.test_size,
            **self.cv_params,
        )
        cv_time = time.time() - start_time

        # Compute metrics (according to `self.metrics`)
        cv_results = self._compute_metrics_and_aggregate(forecasts_cv)

        # Report metrics to Tune
        results = {
            **cv_results,
            "cv_time": cv_time,
        }
        checkpoint_dict = {
            "cross_validation_df": forecasts_cv,
        }

        # Update with metrics/predictions on training data if
        # we passed `fitted` to `StatsForecast.cross_validation`
        if self.cv_params.get("fitted", False):
            train_forecasts_cv = statsforecast.cross_validation_fitted_values()
            train_results = self._compute_metrics_and_aggregate(train_forecasts_cv)
            results["train"] = train_results
            checkpoint_dict["cross_validation_fitted_values_df"] = train_forecasts_cv

        checkpoint = Checkpoint.from_dict(checkpoint_dict)
        session.report(results, checkpoint=checkpoint)

    # __statsforecast_trainer_end__

    def _validate_attributes(self):
        super()._validate_attributes()

        if TRAIN_DATASET_KEY not in self.datasets:
            raise KeyError(
                f"'{TRAIN_DATASET_KEY}' key must be preset in `datasets`. "
                f"Got {list(self.datasets.keys())}"
            )
        scaling_config = self._validate_scaling_config(self.scaling_config)
        if self.parallelize_cv and scaling_config.trainer_resources.get("GPU", 0):
            raise ValueError(
                "`parallelize_cv` cannot be True if there are GPUs assigned to the "
                "trainer."
            )

        not_allowed_keys = ["df", "ray_address", "n_jobs", "models"]
        assert all(
            [key not in self.statsforecast_params for key in not_allowed_keys]
        ), (
            f"`StatsforecastTrainer` will set {not_allowed_keys}, "
            "do not specify these directly."
        )

    def _get_worker_resources(self):
        # Get resource config for this Train worker
        scaling_config = self._validate_scaling_config(self.scaling_config)

        num_workers = scaling_config.num_workers or 0
        assert num_workers == 0  # num_workers is not in scaling config allowed_keys

        trainer_resources = scaling_config.trainer_resources or {"CPU": 1}
        has_gpus = bool(trainer_resources.get("GPU", 0))
        num_cpus = int(trainer_resources.get("CPU", 1))
        return has_gpus, num_cpus

    def _compute_metrics_and_aggregate(self, forecasts_cv: pd.DataFrame) -> Dict:
        # unique_id values are the index of the forecasts dataframe
        unique_ids = forecasts_cv.index.unique()

        cv_aggregates = {}
        for unique_id in unique_ids:
            # If there's only one series in the dataset, we can just report
            # a single metric.
            prefix = f"{unique_id}/" if len(unique_ids) > 1 else ""
            # Calculate metrics separately for each series
            forecasts_for_id = forecasts_cv[forecasts_cv.index == unique_id]
            cutoff_values = forecasts_for_id["cutoff"].unique()

            # Calculate metrics of the predictions of the models fit on
            # each training split
            cv_metrics = defaultdict(list)
            for ct in cutoff_values:
                # Get CV metrics for a specific training window
                # All forecasts made with the same `cutoff` date
                window_df = forecasts_for_id[forecasts_for_id["cutoff"] == ct]
                for metric_name, metric_fn in self.metrics.items():
                    cv_metrics[metric_name].append(
                        metric_fn(
                            window_df["y"], window_df[self.model.__class__.__name__]
                        )
                    )

            # Calculate aggregated metrics (mean, std) across training splits
            for metric_name, metric_vals in cv_metrics.items():
                try:
                    cv_aggregates[f"{prefix}{metric_name}_mean"] = np.nanmean(
                        metric_vals
                    )
                    cv_aggregates[f"{prefix}{metric_name}_std"] = np.nanstd(metric_vals)
                except Exception as e:
                    logger.warning(
                        f"Couldn't calculate aggregate metrics for CV folds! {e}"
                    )
                    cv_aggregates[f"{prefix}{metric_name}_mean"] = np.nan
                    cv_aggregates[f"{prefix}{metric_name}_std"] = np.nan

        return {
            **cv_aggregates,
            "unique_ids": list(unique_ids),
            "cutoff_values": cutoff_values,
        }


if __name__ == "__main__":
    if ray.is_initialized():
        ray.shutdown()
    ray.init(runtime_env={"pip": ["statsforecast"]})

    def get_m5_partition(unique_id: str) -> ray.data.Dataset:
        ds1 = pq.read_table(
            "s3://anonymous@m5-benchmarks/data/train/target.parquet",
            filters=[("item_id", "=", unique_id)],
        )
        Y_df = ds1.to_pandas()
        # StatsForecasts expects specific column names!
        Y_df = Y_df.rename(
            columns={"item_id": "unique_id", "timestamp": "ds", "demand": "y"}
        )
        Y_df["unique_id"] = Y_df["unique_id"].astype(str)
        Y_df["ds"] = pd.to_datetime(Y_df["ds"])
        Y_df = Y_df.dropna()
        constant = 10
        Y_df["y"] += constant
        Y_df = Y_df[Y_df.unique_id == unique_id]
        return ray.data.from_pandas(Y_df)

    ds = get_m5_partition("FOODS_1_001_CA_1")

    from ray.tune.search.optuna import OptunaSearch

    def optuna_search_space(trial) -> Optional[Dict[str, Any]]:
        search_space = {
            AutoARIMA: {},
            ETS: {
                "season_length": [6, 7],
                "model": ["ZNA", "ZZZ"],
            },
        }

        model_type = trial.suggest_categorical("model_cls", list(search_space.keys()))

        # Conditional search space based on the model_type that was chosen
        for param, param_space in search_space[model_type].items():
            trial.suggest_categorical(param, param_space)

        # Return contant params
        return {
            "n_splits": 5,
            "test_size": 1,
            "parallelize_cv": True,
            "freq": "D",
            # "n_parallel_cv_jobs": 5,
        }

    algo = OptunaSearch(space=optuna_search_space, metric="mse_mean", mode="min")
    statsforecast_trainer = StatsforecastTrainer(
        datasets={"train": ds},
        scaling_config=air.ScalingConfig(trainer_resources={"CPU": 4}),
    )
    tuner = tune.Tuner(
        statsforecast_trainer,
        tune_config=tune.TuneConfig(
            metric="mse_mean",
            mode="min",
            search_alg=algo,
            num_samples=5,
        ),
    )
    result_grid = tuner.fit()
    best_result = result_grid.get_best_result()

    print("Best config:", best_result.config)
    print("Best metrics:", best_result.metrics)
