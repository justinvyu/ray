import logging
import os
import pickle
import re
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple

import pyarrow.fs
import pytest

import ray
from ray import train, tune
from ray._private.test_utils import simulate_storage
from ray.air._internal.uri_utils import URI
from ray.air.constants import EXPR_RESULT_FILE
from ray.train._checkpoint import Checkpoint
from ray.train._internal.storage import (
    StorageContext,
    _delete_fs_path,
    _download_from_fs_path,
)
from ray.train.base_trainer import TrainingFailedError
from ray.train.data_parallel_trainer import DataParallelTrainer
from ray.tune.trainable.trainable import _DICT_CHECKPOINT_FILE_NAME


class TestConstants:
    NUM_ITERATIONS = 6  # == num_checkpoints == num_artifacts
    NUM_TRIALS = 2
    NUM_WORKERS = 3

    SCORE_KEY = "score"


@contextmanager
def mock_s3_bucket_uri():
    port = 5002
    region = "us-west-2"
    with simulate_storage("s3", port=port, region=region) as s3_uri:
        import boto3

        s3 = boto3.client(
            "s3", region_name=region, endpoint_url=f"http://localhost:{port}"
        )
        # Bucket name will be autogenerated/unique per test
        bucket_name = URI(s3_uri).name
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": region},
        )
        # Disable server HTTP request logging
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
        yield URI(s3_uri)
        logging.getLogger("werkzeug").setLevel(logging.INFO)


@contextmanager
def dummy_context_manager(*args, **kwargs):
    yield "dummy value"


@pytest.fixture(autouse=True, scope="module")
def ray_start_4_cpus():
    # Make sure to set the env var before calling ray.init()
    ray.init(num_cpus=4)
    yield
    ray.shutdown()


def _create_mock_custom_fs(custom_fs_root_dir: Path) -> pyarrow.fs.FileSystem:
    from fsspec.implementations.dirfs import DirFileSystem
    from fsspec.implementations.local import LocalFileSystem

    custom_fs_root_dir.mkdir(parents=True, exist_ok=True)
    storage_filesystem = pyarrow.fs.PyFileSystem(
        pyarrow.fs.FSSpecHandler(
            DirFileSystem(path=str(custom_fs_root_dir), fs=LocalFileSystem())
        )
    )
    return storage_filesystem


@contextmanager
def _resolve_storage_type(
    storage_path_type: str, tmp_path: Path
) -> Tuple[str, Optional[pyarrow.fs.FileSystem]]:
    storage_path, storage_filesystem = None, None

    context_manager = (
        mock_s3_bucket_uri if storage_path_type == "cloud" else dummy_context_manager
    )

    with context_manager() as cloud_storage_path:
        if storage_path_type == "nfs":
            storage_path = str(tmp_path / "fake_nfs")
        elif storage_path_type == "cloud":
            storage_path = str(cloud_storage_path)
        elif storage_path_type == "custom_fs":
            storage_path = "mock_bucket"
            storage_filesystem = _create_mock_custom_fs(tmp_path / "custom_fs")

        yield storage_path, storage_filesystem


def _get_local_inspect_dir(
    root_local_path: Path,
    storage_path: str,
    storage_local_path: Path,
    storage_filesystem: Optional[pyarrow.fs.FileSystem],
) -> Tuple[Path, str]:
    """Downloads the storage path -> local dir for inspecting contents.

    Returns:
        Tuple: (local_inspect_dir, storage_fs_path), where storage_fs_path
            is the path to the storage path on the filesystem (e.g., prefix stripped).
            This is used to check the correctness of paths returned from `Result`'s,
            since URIs are hard to do comparisons with.
    """
    local_inspect_dir = root_local_path / "inspect"
    if storage_path:
        if storage_filesystem:
            fs, storage_fs_path = storage_filesystem, storage_path
        else:
            fs, storage_fs_path = pyarrow.fs.FileSystem.from_uri(storage_path)
        _download_from_fs_path(
            fs=fs, fs_path=storage_fs_path, local_path=str(local_inspect_dir)
        )
    else:
        fs, storage_fs_path = pyarrow.fs.LocalFileSystem(), str(storage_local_path)
        local_inspect_dir = storage_local_path

    return local_inspect_dir, storage_fs_path


def _get_checkpoint_index(checkpoint_dir_name: str) -> int:
    """Gets the checkpoint index from the checkpoint directory name."""
    return int(checkpoint_dir_name.split("_")[-1])


def _create_checkpoint_shard_filename(rank_str: str) -> str:
    return f"checkpoint_shard-rank={rank_str}.pkl"


def _get_checkpoint_shard_rank(checkpoint_shard_filename: str) -> int:
    """Get the checkpoint shard rank from the filename."""
    pattern = _create_checkpoint_shard_filename(r"(\d+)")
    match = re.search(pattern, checkpoint_shard_filename)
    assert match
    return int(match.group(1))


def train_fn(config):
    in_trainer = config.get("in_trainer", False)
    if in_trainer:
        from ray.air._internal.session import _get_session
        from ray.train._internal.session import _TrainSession

        train_session = _get_session()

        assert isinstance(train_session, _TrainSession)
        assert train_session.storage
        assert train_session.storage.checkpoint_fs_path

        # Check that the working dir for each worker is the shared trial dir.
        assert (
            Path.cwd() == Path(train_session.storage.trial_working_directory).resolve()
        )

    start = 0

    checkpoint = train.get_checkpoint()
    if checkpoint:
        custom_restore_fn = config.get("custom_restore_fn")
        if custom_restore_fn:
            state = custom_restore_fn(checkpoint)
        else:
            with checkpoint.as_directory() as checkpoint_dir:
                with open(os.path.join(checkpoint_dir, "checkpoint.pkl"), "rb") as f:
                    state = pickle.load(f)
        print("Loaded back state from checkpoint:", state)
        start = state["iter"] + 1

    for i in range(start, config.get("num_iterations", 5)):
        time.sleep(config.get("time_per_iter", 0.25))

        metrics = {"iter": i, TestConstants.SCORE_KEY: i}

        # Save an artifact in the local trial dir.
        rank = train.get_context().get_world_rank()
        artifact_file_name = (
            f"artifact-rank={rank}-iter={i}.txt"
            if in_trainer
            else f"artifact-iter={i}.txt"
        )
        with open(artifact_file_name, "w") as f:
            f.write(f"{i}")

        if in_trainer and train.get_context().get_world_rank() in config.get(
            "no_checkpoint_ranks", []
        ):
            train.report(metrics)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                with open(os.path.join(temp_dir, "checkpoint.pkl"), "wb") as f:
                    pickle.dump({"iter": i}, f)

                if in_trainer:
                    checkpoint_file_name = _create_checkpoint_shard_filename(str(rank))
                    with open(os.path.join(temp_dir, checkpoint_file_name), "wb") as f:
                        pickle.dump({"iter": i}, f)

                with config.get("custom_save_fn", dummy_context_manager)(temp_dir):
                    train.report(
                        metrics, checkpoint=Checkpoint.from_directory(temp_dir)
                    )
                # `train.report` should not have deleted this!
                assert os.path.exists(temp_dir)

        if i in config.get("fail_iters", []):
            raise RuntimeError(f"Failing on iter={i}!!")


class ClassTrainable(tune.Trainable):
    """Implement (almost) the same thing as `train_fn` but as a class."""

    def setup(self, config):
        # Save some markers in the trial dir.
        tmp_path = config.get("tmp_path")
        self.fail_markers = {
            i: tmp_path / f"fail_marker_{self.trial_id}_iter={i}"
            for i in config.get("fail_iters", [])
        }
        setup_marker = tmp_path / f"setup_marker_{self.trial_id}"
        if not setup_marker.exists():
            for marker in self.fail_markers.values():
                marker.touch()
            setup_marker.touch()

        self.save_as_dict = config.get("save_checkpoint_as_dict", False)

    def step(self) -> dict:
        if self.iteration in self.fail_markers:
            marker = self.fail_markers[self.iteration]
            if marker.exists():
                marker.unlink()
                raise RuntimeError(f"Failing on iter={self.iteration}")

        # Save an artifact in the local trial dir.
        artifact_file_name = f"artifact-iter={self.iteration}.txt"
        with open(artifact_file_name, "w") as f:
            f.write(f"{self.iteration}")

        return {
            "score": 1,
            "done": self.iteration >= self.config.get("num_iterations") - 1,
            "should_checkpoint": True,
        }

    def save_checkpoint(self, temp_checkpoint_dir) -> str:
        if self.save_as_dict:
            return {"dummy": "data"}
        (Path(temp_checkpoint_dir) / "checkpoint.pkl").write_text("dummy")
        return temp_checkpoint_dir

    def load_checkpoint(self, checkpoint_dict_or_path):
        print("Loading state from:", checkpoint_dict_or_path)
        print("At iteration =", self.iteration)
        if self.save_as_dict:
            assert checkpoint_dict_or_path == {"dummy": "data"}
        else:
            assert (
                Path(checkpoint_dict_or_path) / "checkpoint.pkl"
            ).read_text() == "dummy"


def _resume_from_checkpoint(
    checkpoint: Checkpoint,
    expected_state: dict,
    storage_path: Optional[str] = None,
    storage_filesystem: Optional[pyarrow.fs.FileSystem] = None,
):
    print(f"\nStarting run with `resume_from_checkpoint`: {checkpoint}\n")

    def assert_fn(config):
        checkpoint_to_check = train.get_checkpoint()
        with checkpoint_to_check.as_directory() as checkpoint_dir:
            with open(os.path.join(checkpoint_dir, "checkpoint.pkl"), "rb") as f:
                state = pickle.load(f)

        print("Loaded state from `resume_from_checkpoint`:", state)
        print("Expected state:", expected_state)
        assert state == expected_state, (state, expected_state)

        dummy_ckpt = tempfile.mkdtemp()
        with open(os.path.join(dummy_ckpt, "dummy.txt"), "w") as f:
            f.write("data")
        train.report({"dummy": 1}, checkpoint=Checkpoint.from_directory(dummy_ckpt))

    trainer = DataParallelTrainer(
        assert_fn,
        scaling_config=train.ScalingConfig(num_workers=2),
        run_config=train.RunConfig(
            name="test_resume_from_checkpoint",
            storage_path=storage_path,
            storage_filesystem=storage_filesystem,
        ),
        resume_from_checkpoint=checkpoint,
    )
    result = trainer.fit()

    # Make sure that the checkpoint indexing starts from scratch.
    assert Path(
        result.checkpoint.path
    ).name == StorageContext._make_checkpoint_dir_name(0)

    # Clean up this run's experiment directory immediately after.
    _delete_fs_path(result.filesystem, Path(result.path).parent.as_posix())


def _assert_storage_contents(
    local_inspect_dir: Path,
    exp_name: str,
    checkpoint_config: train.CheckpointConfig,
    trainable_name: str,
    test_trainer: bool,
    no_checkpoint_ranks: List[int] = None,
    constants: type = TestConstants,
):
    no_checkpoint_ranks = no_checkpoint_ranks or []

    # Second, inspect the contents of the storage path
    storage_path_ls = list(local_inspect_dir.glob("*"))
    assert len(storage_path_ls) == 1  # Only expect 1 experiment dir
    exp_dir = storage_path_ls[0]
    assert exp_dir.name == exp_name

    # Files synced by the driver
    assert len(list(exp_dir.glob("tuner.pkl"))) == 1
    if test_trainer:
        assert len(list(exp_dir.glob("trainer.pkl"))) == 1
    # 2 copies of these files:
    # 1 for the initial run, and 1 for the manually restored run.
    assert len(list(exp_dir.glob("basic-variant-state-*"))) == 2
    assert len(list(exp_dir.glob("experiment_state-*"))) == 2

    # Files synced by the worker
    assert (
        len(list(exp_dir.glob(f"{trainable_name}*"))) == 1
        if test_trainer
        else constants.NUM_TRIALS
    )
    for trial_dir in exp_dir.glob(f"{trainable_name}*"):
        # If set, expect num_to_keep. Otherwise, expect to see all of them.
        expected_num_checkpoints = (
            checkpoint_config.num_to_keep or constants.NUM_ITERATIONS
        )

        assert len(list(trial_dir.glob("checkpoint_*"))) == expected_num_checkpoints
        checkpoint_idxs = sorted(
            [
                _get_checkpoint_index(checkpoint_dir.name)
                for checkpoint_dir in trial_dir.glob("checkpoint_*")
            ]
        )
        # Ex: If num_to_keep=2 out of 6 total checkpoints,
        # expect checkpoint_004 and checkpoint_005.
        assert checkpoint_idxs == list(
            range(
                constants.NUM_ITERATIONS - expected_num_checkpoints,
                constants.NUM_ITERATIONS,
            )
        )

        for checkpoint_dir in trial_dir.glob("checkpoint_*"):
            # 1 shared checkpoint.pkl file, written by the trainable / all workers.
            assert (
                len(list(checkpoint_dir.glob("checkpoint.pkl"))) == 1
                # NOTE: Dict checkpoint is only for the ClassTrainable.
                or len(list(checkpoint_dir.glob(_DICT_CHECKPOINT_FILE_NAME))) == 1
            )
            if test_trainer:
                # 1 checkpoint shard per worker.
                # Unless the worker did not report a checkpoint (no_checkpoint_ranks).
                assert {
                    _get_checkpoint_shard_rank(checkpoint_shard.name)
                    for checkpoint_shard in checkpoint_dir.glob(
                        "checkpoint_shard-*.pkl"
                    )
                } == {
                    i
                    for i in range(constants.NUM_WORKERS)
                    if i not in no_checkpoint_ranks
                }

        if test_trainer:
            expected_num_artifacts = constants.NUM_ITERATIONS * constants.NUM_WORKERS
        else:
            expected_num_artifacts = constants.NUM_ITERATIONS
        assert len(list(trial_dir.glob("artifact-*"))) == expected_num_artifacts

        # NOTE: This result file is synced by the driver.
        assert len(list(trial_dir.glob(EXPR_RESULT_FILE))) == 1


@pytest.mark.parametrize("trainable", [train_fn, ClassTrainable])
@pytest.mark.parametrize("storage_path_type", [None, "nfs", "cloud", "custom_fs"])
@pytest.mark.parametrize(
    "checkpoint_config",
    [train.CheckpointConfig(), train.CheckpointConfig(num_to_keep=2)],
)
def test_tuner(
    monkeypatch,
    tmp_path,
    trainable,
    storage_path_type,
    checkpoint_config: train.CheckpointConfig,
):
    """End-to-end test that the new persistence mode works with the Tuner API.
    This test covers many `storage_path_type` options:
    - storage_path=None --> save locally to the default local path (e.g., ~/ray_results)
    - storage_path="nfs" --> save locally to a fake NFS path
    - storage_path="cloud" --> save to a mock S3 bucket
    - storage_path="custom_fs" --> save to a custom pyarrow filesystem
        - The custom fs is a local filesystem that appends a path prefix to every path.

    This is the expected output at the storage path:

    {storage_path}/{exp_name}
    ├── tuner.pkl                   <- Driver artifacts (global experiment state)
    ├── basic-variant-state.json
    ├── experiment_state.json
    ├── train_fn_a2b9e_00000_0_...
    │   ├── artifact-iter=0.txt     <- Trial artifacts
    │   ├── ...
    │   ├── checkpoint_000000       <- Trial checkpoints
    │   │   └── checkpoint.pkl
    │   ├── ...
    │   ├── events.out.tfevents...  <- Driver artifacts (trial results)
    │   ├── params.json
    │   ├── params.pkl
    │   ├── progress.csv
    │   └── result.json
    └── train_fn_a2b9e_00001_1_...
        └── ...                     <- Same as above
    """
    # Set the cache dir to some temp directory
    LOCAL_CACHE_DIR = tmp_path / "ray_results"
    monkeypatch.setenv("RAY_AIR_LOCAL_CACHE_DIR", str(LOCAL_CACHE_DIR))

    exp_name = "simple_persistence_test"

    with _resolve_storage_type(storage_path_type, tmp_path) as (
        storage_path,
        storage_filesystem,
    ):
        tuner = tune.Tuner(
            trainable,
            param_space={
                "num_iterations": TestConstants.NUM_ITERATIONS,
                "fail_iters": [2, 4],
                # NOTE: This param is only used in the ClassTrainable.
                "save_checkpoint_as_dict": tune.grid_search([True, False]),
                "tmp_path": tmp_path,
            },
            run_config=train.RunConfig(
                storage_path=storage_path,
                storage_filesystem=storage_filesystem,
                name=exp_name,
                verbose=0,
                failure_config=train.FailureConfig(max_failures=1),
                checkpoint_config=checkpoint_config,
                sync_config=train.SyncConfig(sync_artifacts=True),
            ),
            # 2 samples (from the grid search). Run 1 at at time to test actor reuse
            tune_config=tune.TuneConfig(num_samples=1, max_concurrent_trials=1),
        )
        result_grid = tuner.fit()
        assert result_grid.errors

        if storage_path:
            shutil.rmtree(LOCAL_CACHE_DIR, ignore_errors=True)

        restored_tuner = tune.Tuner.restore(
            path=str(URI(storage_path or str(LOCAL_CACHE_DIR)) / exp_name),
            trainable=trainable,
            storage_filesystem=storage_filesystem,
            resume_errored=True,
        )
        result_grid = restored_tuner.fit()
        assert not result_grid.errors

        local_inspect_dir, storage_fs_path = _get_local_inspect_dir(
            root_local_path=tmp_path,
            storage_path=storage_path,
            storage_local_path=LOCAL_CACHE_DIR,
            storage_filesystem=storage_filesystem,
        )

    # First, check that the ResultGrid returns the correct paths.
    print(result_grid)
    experiment_fs_path = result_grid.experiment_path
    assert isinstance(result_grid.filesystem, pyarrow.fs.FileSystem), result_grid
    assert experiment_fs_path == os.path.join(storage_fs_path, exp_name)
    assert len(result_grid) == TestConstants.NUM_TRIALS
    for result in result_grid:
        trial_fs_path = result.path
        assert isinstance(result.filesystem, pyarrow.fs.FileSystem), result
        assert trial_fs_path.startswith(experiment_fs_path)
        for checkpoint, _ in result.best_checkpoints:
            assert checkpoint.path.startswith(trial_fs_path)

    # Next, inspect the storage path contents.
    _assert_storage_contents(
        local_inspect_dir,
        exp_name,
        checkpoint_config,
        trainable_name=trainable.__name__,
        test_trainer=False,
    )


@pytest.mark.parametrize("storage_path_type", [None, "nfs", "cloud", "custom_fs"])
@pytest.mark.parametrize(
    "checkpoint_config",
    [
        train.CheckpointConfig(),
        train.CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute=TestConstants.SCORE_KEY,
            checkpoint_score_order="max",
        ),
    ],
)
def test_trainer(
    tmp_path, monkeypatch, storage_path_type, checkpoint_config: train.CheckpointConfig
):
    """Same end-to-end test as `test_tuner`, but also includes a
    `DataParallelTrainer(resume_from_checkpoint)` test at the end.

    {storage_path}/{exp_name}
    ├── experiment_state-2023-07-28_10-00-38.json       <- Initial exp state
    ├── basic-variant-state-2023-07-28_10-00-38.json
    ├── experiment_state-2023-07-28_10-01-38.json       <- Restored exp state
    ├── basic-variant-state-2023-07-28_10-01-38.json
    ├── trainer.pkl
    ├── tuner.pkl
    └── DataParallelTrainer_46367_00000_0_...
        ├── events.out.tfevents...
        ├── params.json
        ├── params.pkl
        ├── progress.csv
        ├── result.json
        ├── checkpoint_000000
        │   ├── checkpoint.pkl                    <- Shared checkpoint file
        │   ├── checkpoint_shard-rank=0.pkl       <- Worker checkpoint shards
        │   └── checkpoint_shard-rank=1.pkl
        ├── ...
        ├── artifact-rank=0-iter=0.txt            <- Worker artifacts
        ├── artifact-rank=1-iter=0.txt
        ├── ...
        ├── artifact-rank=0-iter=1.txt
        ├── artifact-rank=1-iter=1.txt
        └── ...
    """
    LOCAL_CACHE_DIR = tmp_path / "ray_results"
    monkeypatch.setenv("RAY_AIR_LOCAL_CACHE_DIR", str(LOCAL_CACHE_DIR))
    exp_name = "trainer_new_persistence"
    no_checkpoint_ranks = [0]

    with _resolve_storage_type(storage_path_type, tmp_path) as (
        storage_path,
        storage_filesystem,
    ):
        trainer = DataParallelTrainer(
            train_fn,
            train_loop_config={
                "in_trainer": True,
                "num_iterations": TestConstants.NUM_ITERATIONS,
                "fail_iters": [2, 4],
                # Test that global rank 0 is not required to checkpoint.
                "no_checkpoint_ranks": no_checkpoint_ranks,
            },
            scaling_config=train.ScalingConfig(num_workers=TestConstants.NUM_WORKERS),
            run_config=train.RunConfig(
                storage_path=storage_path,
                storage_filesystem=storage_filesystem,
                name=exp_name,
                verbose=0,
                checkpoint_config=checkpoint_config,
                failure_config=train.FailureConfig(max_failures=1),
                sync_config=train.SyncConfig(sync_artifacts=True),
            ),
        )
        print("\nStarting initial run.\n")
        with pytest.raises(TrainingFailedError):
            result = trainer.fit()

        print("\nStarting manually restored run.\n")
        restored_trainer = DataParallelTrainer.restore(
            path=str(URI(storage_path or str(LOCAL_CACHE_DIR)) / exp_name),
            storage_filesystem=storage_filesystem,
        )
        result = restored_trainer.fit()

        _resume_from_checkpoint(
            result.checkpoint,
            expected_state={"iter": TestConstants.NUM_ITERATIONS - 1},
        )

        local_inspect_dir, storage_fs_path = _get_local_inspect_dir(
            root_local_path=tmp_path,
            storage_path=storage_path,
            storage_local_path=LOCAL_CACHE_DIR,
            storage_filesystem=storage_filesystem,
        )

    # First, inspect that the result object returns the correct paths.
    print(result)
    trial_fs_path = result.path
    assert trial_fs_path.startswith(storage_fs_path)
    for checkpoint, _ in result.best_checkpoints:
        assert checkpoint.path.startswith(trial_fs_path)

    _assert_storage_contents(
        local_inspect_dir,
        exp_name,
        checkpoint_config,
        trainable_name="DataParallelTrainer",
        test_trainer=True,
        no_checkpoint_ranks=no_checkpoint_ranks,
    )


def test_local_dir(tmp_path):
    """Test that local_dir can do the same job as `RAY_AIR_LOCAL_CACHE_DIR`."""

    def train_fn(config):
        from ray.train._internal.session import get_session

        assert get_session().storage.storage_local_path == str(tmp_path)

    tune.run(train_fn, local_dir=str(tmp_path))

    results = tune.Tuner(
        train_fn, run_config=train.RunConfig(local_dir=str(tmp_path))
    ).fit()
    assert not results.errors

    trainer = DataParallelTrainer(
        train_fn,
        scaling_config=train.ScalingConfig(num_workers=2),
        run_config=train.RunConfig(local_dir=str(tmp_path)),
    )
    trainer.fit()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
