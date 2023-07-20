import dataclasses
from typing import Optional

import pyarrow.fs

from ray.air._internal.uri_utils import is_uri
from ray.train import RunConfig
from ray.tune.syncer import Syncer, SyncConfig, _DefaultSyncer
from ray.tune.result import _get_defaults_results_dir


class StorageContext:
    def __init__(self, run_config: RunConfig):
        self.storage_path: str = run_config.storage_path
        self.storage_cache_dir: str = _get_defaults_results_dir()
        self.experiment_dir_name: str = run_config.name
        self.sync_config: SyncConfig = dataclasses.replace(run_config.sync_config)

        if run_config.storage_filesystem:
            # Custom pyarrow filesystem
            self.storage_filesystem = run_config.storage_filesystem
            if is_uri(self.storage_path):
                raise ValueError("TODO")
            self.storage_path_on_filesystem = self.storage_path
        else:
            (
                self.storage_filesystem,
                self.storage_path_on_filesystem,
            ) = pyarrow.fs.FileSystem.from_uri(self.storage_path)
            print(f"Auto-detected storage filesystem.")

        self.syncer: Optional[Syncer] = (
            None
            if self.storage_path == self.storage_cache_dir
            else _DefaultSyncer(
                sync_period=self.sync_config.sync_period,
                sync_timeout=self.sync_config.sync_timeout,
                storage_filesystem=self.storage_filesystem,
            )
        )

        self._create_validation_file()
        self._check_validation_file()

    def _create_validation_file(self):
        valid_file = self.storage_prefix + "/_valid"
        self.storage_filesystem.create_dir(self.storage_prefix)
        with self.storage_filesystem.open_output_stream(valid_file):
            pass

    def _check_validation_file(self):
        valid_file = self.storage_prefix + "/_valid"
        valid = self.storage_filesystem.get_file_info([valid_file])[0]
        if valid.type == pyarrow.fs.FileType.NotFound:
            raise RuntimeError(
                "Unable to initialize storage: {} file created during init not found. "
                "Check that configured cluster storage path is readable from all "
                "worker nodes of the cluster.".format(valid_file)
            )

    @property
    def experiment_dir(self) -> str:
        pass

    # @property
    # def experiment_cache_dir(self) -> str:
    #     pass

    @property
    def trial_dir(self):
        pass

    # @property
    # def trial_cache_path(self):
    #     pass

    def construct_checkpoint_path(self, checkpoint_dir_name: str) -> str:
        pass


# Maybe have it be a global variable??
# To communicate from trainable -> "Trainer", rather than pipe it from the BaseTrainer
# The BaseTrainer could have a different RunConfig...
_storage_context: Optional[StorageContext] = None


def init_storage_context(storage_context: Optional[StorageContext]):
    global _storage_context
    _storage_context = storage_context


def get_storage_context() -> Optional[StorageContext]:
    return _storage_context
