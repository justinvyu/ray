import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsforecast import StatsForecast

from ray.air import Checkpoint, session
from ray.air.config import RunConfig, ScalingConfig
from ray.train.constants import TRAIN_DATASET_KEY
from ray.train.trainer import BaseTrainer, GenDataset

if TYPE_CHECKING:
    from ray.data.preprocessor import Preprocessor

logger = logging.getLogger(__name__)

ArrayType = Union[pd.DataFrame, np.ndarray]
MetricType = Union[str, Callable[[ArrayType, ArrayType], float]]
ScoringType = Union[MetricType, Iterable[MetricType], Dict[str, MetricType]]


class StatsforecastTrainer(BaseTrainer):
    def __init__(
        self,
        *,
        model_cls_and_params=None,
        model_cls=None,
        # model_params=None,
        datasets: Dict[str, GenDataset] = None,
        freq: str = "D",
        metrics: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        return_train_forecasts_cv: bool = False,
        parallelize_cv: bool = True,
        set_estimator_cpus: bool = True,
        scaling_config: Optional[ScalingConfig] = None,
        run_config: Optional[RunConfig] = None,
        preprocessor: Optional["Preprocessor"] = None,
        **model_params,
    ):
        if model_params.pop("resume_from_checkpoint", None):
            raise AttributeError(
                "StatsforecastTrainer does not support resuming from checkpoints. "
                "Remove the `resume_from_checkpoint` argument."
            )

        # TODO(justinvyu): remove this tempororary hack
        # (Tune search space not expressive enough?)
        if model_cls_and_params:
            assert not model_cls and not model_params, (
                "Specify statsforecast model class and params either as a tuple "
                "`(model_cls, model_params)` or separately."
            )
            self.model_cls, self.model_params = model_cls_and_params
        else:
            self.model_cls = model_cls
            self.model_params = model_params
        self.freq = freq
        self.metrics = metrics or {
            "mse": mean_squared_error,
            "mae": mean_absolute_error,
        }
        self.params = params or {}
        # self.fit_params = fit_params
        self.n_splits = n_splits
        self.test_size = test_size
        # self.cv_params = cv_params or {"n_windows": 5}
        self.parallelize_cv = parallelize_cv
        self.set_estimator_cpus = set_estimator_cpus
        self.return_train_forecasts_cv = return_train_forecasts_cv
        super().__init__(
            scaling_config=scaling_config,
            run_config=run_config,
            datasets=datasets,
            preprocessor=preprocessor,
            resume_from_checkpoint=None,
        )

    def _validate_attributes(self):
        super()._validate_attributes()

        if self.params is not None and not isinstance(self.params, dict):
            raise ValueError(f"`params` must be a dict or None, got '{self.params}'")

        if not isinstance(self.return_train_forecasts_cv, bool):
            raise ValueError(
                f"`return_train_forecasts_cv` must be a boolean, got "
                f"'{self.return_train_score_cv}'"
            )

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

    def _compute_metrics_and_aggregate(self, forecasts_cv: pd.DataFrame) -> Dict:
        # unique_id values are the index of the forecasts dataframe
        unique_ids = forecasts_cv.index.unique()

        cv_aggregates = {}
        for unique_id in unique_ids:
            prefix = f"{unique_id}/" if len(unique_ids) > 1 else ""
            # Calculate metrics separately for each series
            forecasts_for_id = forecasts_cv[forecasts_cv.index == unique_id]
            cutoff_values = forecasts_for_id["cutoff"].unique()

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
            "unique_ids": list(unique_ids),
            # **cv_metrics,
            **cv_aggregates,
            "cutoff_values": cutoff_values,
        }

    def setup(self) -> None:
        # Construct the model from the class/params passed in
        self.model = self.model_cls(**self.model_params)

    def _get_worker_resources(self):
        # Get resource config for this Train worker
        scaling_config = self._validate_scaling_config(self.scaling_config)

        num_workers = scaling_config.num_workers or 0
        assert num_workers == 0  # num_workers is not in scaling config allowed_keys

        trainer_resources = scaling_config.trainer_resources or {"CPU": 1}
        has_gpus = bool(trainer_resources.get("GPU", 0))
        num_cpus = int(trainer_resources.get("CPU", 1))
        return has_gpus, num_cpus

    def training_loop(self) -> None:
        # Get training dataset
        datasets = self._get_datasets()
        Y_train_df = datasets.pop(TRAIN_DATASET_KEY)

        # Assign resources
        has_gpus, num_cpus = self._get_worker_resources()

        assert not (has_gpus and self.parallelize_cv)

        if self.parallelize_cv:
            # Connect to the existing ray cluster via `auto`
            parallelism_kwargs = {"ray_address": "auto", "n_jobs": num_cpus}
        else:
            parallelism_kwargs = {"n_jobs": 1}

        # Initialize statsforecast with the model, using Ray as its parallel backend
        statsforecast = StatsForecast(
            df=Y_train_df,
            models=[self.model],
            freq=self.freq,
            **parallelism_kwargs,
            # TODO(justinvyu): add more kwargs for user here?
        )

        # TODO(justinvyu): Fit model and save it? Should this be optional?
        # Wasted computation if you need to retrain on a new dataframe with new points
        # self.model.fit()

        # Perform temporal cross validation
        # Set args in statsforecast.cross_validation to match the splitting behavior of
        # sklearn's TimeSeriesSplit
        # Configurations: `n_splits`, `test_size`
        # TODO(justinvyu): allow for sliding eval (currently only rolling)

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
            fitted=self.return_train_forecasts_cv,
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
        # `self.return_train_forecasts_cv`
        if self.return_train_forecasts_cv:
            train_forecasts_cv = statsforecast.cross_validation_fitted_values()
            train_results = self._compute_metrics_and_aggregate(train_forecasts_cv)
            results["train"] = train_results
            checkpoint_dict["cross_validation_fitted_values_df"] = train_forecasts_cv

        checkpoint = Checkpoint.from_dict(checkpoint_dict)
        session.report(results, checkpoint=checkpoint)
