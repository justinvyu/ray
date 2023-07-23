from contextlib import contextmanager
import os
import pickle
import pytest
import tempfile
import time

from ray import air, train, tune
from ray.air.constants import EXPR_RESULT_FILE
from ray.air.tests.test_checkpoints import mock_s3_bucket_uri
from ray.air._internal.remote_storage import download_from_uri


@contextmanager
def dummy_context_manager():
    yield "dummy value"


@pytest.fixture(autouse=True)
def enable_new_persistence_mode(monkeypatch):
    monkeypatch.setenv("RAY_AIR_NEW_PERSISTENCE_MODE", "1")
    yield
    monkeypatch.setenv("RAY_AIR_NEW_PERSISTENCE_MODE", "0")


def train_fn(config):
    start = 0

    checkpoint = train.get_context().get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            with open(os.path.join(checkpoint_dir, "dummy.pkl"), "rb") as f:
                state = pickle.load(f)
        print("Loaded back state from checkpoint:", state)
        start = state["iter"] + 1

    for i in range(start, config.get("num_iterations", 5)):
        time.sleep(0.5)

        with open(f"artifact-{i}.txt", "w") as f:
            f.write(f"{i}")

        temp_dir = tempfile.mkdtemp()
        with open(os.path.join(temp_dir, "dummy.pkl"), "wb") as f:
            pickle.dump({"iter": i}, f)

        train.report({"iter": i}, checkpoint=air.Checkpoint.from_directory(temp_dir))

        if i in config.get("fail_iters", []):
            raise RuntimeError(f"Failing at iter={i}")


@pytest.mark.parametrize("storage_path_type", [None, "nfs", "cloud", "custom_fs"])
def test_tuner(monkeypatch, storage_path_type, tmp_path):
    """End-to-end test that the new persistence mode works with the Tuner API.

    This test covers many `storage_path_type` options:
    - storage_path=None --> save locally to the default cache path
    - storage_path="nfs" --> save locally to a fake NFS path
    - storage_path="cloud" --> save to a mock S3 bucket
    - storage_path="custom_fs" --> save to a custom pyarrow filesystem
        - The custom fs is a local filesystem that appends a path prefix to every path.

    For each trial, the test training code will:
    - Fail at iter=2, then auto-retry due to max_failures=1
    - Fail at iter=4, then terminate (out of retries).
    - Restore and finish successfully with a final report at iter=5.
    """

    # Set the cache dir to some temp directory
    LOCAL_CACHE_DIR = tmp_path / "ray_results"
    monkeypatch.setenv("RAY_AIR_LOCAL_CACHE_DIR", str(LOCAL_CACHE_DIR))

    context_manager = (
        mock_s3_bucket_uri if storage_path_type == "cloud" else dummy_context_manager
    )

    # TODO(justinvyu): custom_fs doesn't work at the moment.
    # Splitting off support for this for a follow-up PR, after the new Checkpoint
    # class is introduced.
    if storage_path_type == "custom_fs":
        return

    exp_name = "simple_persistence_test"
    should_test_restore = True

    with context_manager() as cloud_storage_path:
        storage_filesystem = None
        if storage_path_type is None:
            storage_path = None
        elif storage_path_type == "nfs":
            storage_path = str(tmp_path / "fake_nfs")
        elif storage_path_type == "cloud":
            storage_path = str(cloud_storage_path)
        elif storage_path_type == "custom_fs":
            from fsspec.implementations.dirfs import DirFileSystem
            from fsspec.implementations.local import LocalFileSystem

            import pyarrow.fs

            storage_path = "mock_bucket"
            storage_filesystem = pyarrow.fs.PyFileSystem(
                pyarrow.fs.FSSpecHandler(
                    DirFileSystem(
                        path=str(tmp_path / "custom_fs"), fs=LocalFileSystem()
                    )
                )
            )

        NUM_ITERATIONS = 6  # == num_checkpoints == num_artifacts
        NUM_TRIALS = 2

        param_space = {
            "num_iterations": NUM_ITERATIONS,
            "fail_iters": [2, 4] if should_test_restore else [],
        }

        tuner = tune.Tuner(
            train_fn,
            param_space=param_space,
            run_config=train.RunConfig(
                storage_path=storage_path,
                storage_filesystem=storage_filesystem,
                name="simple_persistence_test",
                verbose=0,
                failure_config=train.FailureConfig(max_failures=1),
            ),
            # 2 samples, running 1 at at time to test with actor reuse
            tune_config=tune.TuneConfig(
                num_samples=NUM_TRIALS, max_concurrent_trials=1
            ),
        )
        results = tuner.fit()

        if should_test_restore:
            print("Manual experiment restore from:", results.experiment_path)
            tuner = tune.Tuner.restore(
                results.experiment_path, trainable=train_fn, resume_errored=True
            )
            results = tuner.fit()

        local_inspect_dir = tmp_path / "inspect"
        if storage_path:
            download_from_uri(
                storage_path,
                str(local_inspect_dir),
                source_filesystem=storage_filesystem,
            )
        else:
            local_inspect_dir = LOCAL_CACHE_DIR

    assert len(list(local_inspect_dir.glob("*"))) == 1  # Only expect 1 experiment dir
    exp_dir = local_inspect_dir / exp_name

    # Files synced by the driver
    assert len(list(exp_dir.glob("basic-variant-state-*"))) == (
        2 if should_test_restore else 1
    )
    assert len(list(exp_dir.glob("experiment_state-*"))) == (
        2 if should_test_restore else 1
    )
    assert len(list(exp_dir.glob("tuner.pkl"))) == 1

    # Files synced by the Trainable
    assert len(list(exp_dir.glob("train_*"))) == NUM_TRIALS
    for trial_dir in exp_dir.glob("train_*"):
        assert len(list(trial_dir.glob("checkpoint_*"))) == NUM_ITERATIONS
        assert len(list(trial_dir.glob("checkpoint_*/dummy.pkl"))) == NUM_ITERATIONS
        assert len(list(trial_dir.glob("artifact-*"))) == NUM_ITERATIONS
        assert len(list(trial_dir.glob(EXPR_RESULT_FILE))) == 1
