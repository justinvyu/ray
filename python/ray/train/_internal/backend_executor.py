import logging
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar

import ray
from ray.data import Dataset
from ray._private.ray_constants import env_integer
from ray.air.config import CheckpointConfig
from ray.exceptions import RayActorError
from ray.train import DataConfig
from ray.air.checkpoint import Checkpoint
from ray.train._internal.session import (
    TrainingResult,
    TrialInfo,
    get_session,
    init_session,
    shutdown_session,
)
from ray.train._internal.storage import _use_storage_context, get_storage_context
from ray.train._internal.utils import check_for_failure
from ray.train._internal.worker_group import WorkerGroup
from ray.train.backend import BackendConfig
from ray.train.constants import (
    ENABLE_DETAILED_AUTOFILLED_METRICS_ENV,
    ENABLE_SHARE_CUDA_VISIBLE_DEVICES_ENV,
    TRAIN_ENABLE_WORKER_SPREAD_ENV,
    TRAIN_PLACEMENT_GROUP_TIMEOUT_S_ENV,
    DISABLE_LAZY_CHECKPOINTING_ENV,
)
from ray.util.placement_group import get_current_placement_group, remove_placement_group

T = TypeVar("T")

logger = logging.getLogger(__name__)


class TrainBackendError(Exception):
    """Errors with BackendExecutor that should not be exposed to user."""


class TrainingWorkerError(Exception):
    """Raised if a worker fails during training."""


class BackendExecutor:
    """Main execution class for training backends.

    This class holds a worker group and is responsible for executing the
    training function on the workers, and collecting intermediate results
    from ``session.report()``.

    Args:
        backend_config: The configurations for this
            specific backend.
        num_workers: Number of workers to use for training.
        num_cpus_per_worker: Number of CPUs to use per worker.
        num_gpus_per_worker: Number of GPUs to use per worker.
        additional_resources_per_worker (Optional[Dict[str, float]]):
            Dictionary specifying the extra resources that will be
            requested for each worker in addition to ``num_cpus_per_worker``
            and ``num_gpus_per_worker``.
        max_retries: Number of retries when Ray actors fail.
            Defaults to 3. Set to -1 for unlimited retries.
    """

    def __init__(
        self,
        backend_config: BackendConfig,
        # TODO(xwjiang): Legacy Ray Train trainer clean up!
        trial_info: Optional[TrialInfo] = None,
        num_workers: int = 1,
        num_cpus_per_worker: float = 1,
        num_gpus_per_worker: float = 0,
        additional_resources_per_worker: Optional[Dict[str, float]] = None,
        max_retries: int = 3,
        checkpoint_config: Optional[CheckpointConfig] = None,
    ):
        self._backend_config = backend_config
        self._backend = backend_config.backend_cls()
        self._num_workers = num_workers
        self._num_cpus_per_worker = num_cpus_per_worker
        self._num_gpus_per_worker = num_gpus_per_worker
        self._additional_resources_per_worker = additional_resources_per_worker
        self._max_failures = max_retries
        if self._max_failures < 0:
            self._max_failures = float("inf")
        self._num_failures = 0
        self._last_failure = None
        self._initialization_hook = None
        self._placement_group = None

        self._trial_info = trial_info

        self.worker_group = InactiveWorkerGroup()
        self.dataset_shards = None

        self._checkpoint_keep_all_ranks = (
            checkpoint_config and checkpoint_config._checkpoint_keep_all_ranks
        )
        self._checkpoint_upload_from_workers = (
            checkpoint_config and checkpoint_config._checkpoint_upload_from_workers
        )

    def start(
        self,
        initialization_hook: Optional[Callable[[], None]] = None,
        train_cls: Optional[Type] = None,
        train_cls_args: Optional[Tuple] = None,
        train_cls_kwargs: Optional[Dict] = None,
    ):
        """Starts the worker group."""
        self._create_placement_group()
        placement_group = self._placement_group or "default"
        self.worker_group = WorkerGroup(
            num_workers=self._num_workers,
            num_cpus_per_worker=self._num_cpus_per_worker,
            num_gpus_per_worker=self._num_gpus_per_worker,
            additional_resources_per_worker=self._additional_resources_per_worker,
            actor_cls=train_cls,
            actor_cls_args=train_cls_args,
            actor_cls_kwargs=train_cls_kwargs,
            placement_group=placement_group,
        )
        # Hack to avoid OOMs.
        # This is just a temporary solution for Train loading entire checkpoints
        # into memory by ensuring that the rank 0 worker is on the same node as
        # trainable, thus allowing for lazy checkpoint transfer to be used.
        # See https://github.com/ray-project/ray/issues/33073
        # for more context.
        # TODO remove
        if self._trial_info and self._trial_info.driver_ip:
            self.worker_group._move_workers_with_ip_to_front(self._trial_info.driver_ip)

        worker_locs = [
            f"{w.metadata.pid} ({w.metadata.node_ip})"
            for w in self.worker_group.workers
        ]
        logger.info(f"Starting distributed worker processes: {worker_locs}")

        try:
            if initialization_hook:
                self._initialization_hook = initialization_hook
                self.worker_group.execute(initialization_hook)

            share_cuda_visible_devices_enabled = bool(
                env_integer(
                    ENABLE_SHARE_CUDA_VISIBLE_DEVICES_ENV,
                    self._backend.share_cuda_visible_devices,
                )
            )

            if self._num_gpus_per_worker > 0 and share_cuda_visible_devices_enabled:
                self._share_cuda_visible_devices()
            self._backend.on_start(self.worker_group, self._backend_config)
        except RayActorError as exc:
            logger.exception(str(exc))
            logger.warning(
                "Failure occurred during startup. Restarting all workers and "
                "attempting to startup again."
            )
            self._increment_failures()
            self._restart()

    def _create_placement_group(self):
        """Creates a placement group if it does not exist.

        If a placement group is already detected (Tune) this will be a no-op.

        By default the placement group will be created with PACK strategy.
        This is optimized for colocating GPUs on a minimal number of nodes.
        This behavior can be overridden to use the SPREAD strategy by defining
        ``TRAIN_ENABLE_WORKER_SPREAD_ENV``

        If a placement group is created it will be stored as
        self._placement_group.
        """
        current_placement_group = get_current_placement_group()
        worker = ray._private.worker.global_worker
        should_capture_child_tasks_in_placement_group = (
            worker.should_capture_child_tasks_in_placement_group
        )
        should_create_placement_group = (
            current_placement_group is None
            or not should_capture_child_tasks_in_placement_group
        )

        if should_create_placement_group:
            additional_resources_per_worker = (
                self._additional_resources_per_worker or {}
            )
            bundle = {
                "CPU": self._num_cpus_per_worker,
                "GPU": self._num_gpus_per_worker,
                **additional_resources_per_worker,
            }
            bundles = [bundle.copy() for _ in range(self._num_workers)]

            use_spread = bool(env_integer(TRAIN_ENABLE_WORKER_SPREAD_ENV, 0))
            strategy = "SPREAD" if use_spread else "PACK"

            placement_group = ray.util.placement_group(bundles, strategy=strategy)
            logger.debug("Waiting for placement group to start.")
            timeout = env_integer(TRAIN_PLACEMENT_GROUP_TIMEOUT_S_ENV, 100)
            ready, _ = ray.wait([placement_group.ready()], timeout=timeout)
            if ready:
                logger.debug("Placement group has started.")
            else:
                raise TimeoutError(
                    "Placement group creation timed out. Make sure your "
                    "cluster either has enough resources or use an "
                    "autoscaling cluster. If you are running on a cluster, "
                    "make sure you specify an address in `ray.init()`, for example, "
                    '`ray.init("auto")`. You can also increase the timeout by setting '
                    "the TRAIN_PLACEMENT_GROUP_TIMEOUT_S environment variable. "
                    "Current resources available: {}, resources requested by the "
                    "placement group: {}".format(
                        ray.available_resources(), placement_group.bundle_specs
                    )
                )
            self._placement_group = placement_group

    def _share_cuda_visible_devices(self):
        """Sets CUDA_VISIBLE_DEVICES on all workers.

        For each worker, CUDA_VISIBLE_DEVICES will be set to the GPU IDs
        visible to all workers on that worker's node.

        This allows GPU workers on the same node to communicate with one
        another.

        Example:

            Setup:
            - Node1:
                - Worker1: {0, 1}
                - Worker2: {2, 3}
            - Node2:
                - Worker3: {0, 1}

            CUDA_VISIBLE_DEVICES:
            - Worker1: "0,1,2,3"
            - Worker2: "0,1,2,3"
            - Worker2: "0,1"

        """

        node_ids_and_gpu_ids = [
            (w.metadata.node_id, w.metadata.gpu_ids) for w in self.worker_group.workers
        ]

        node_id_to_worker_id = defaultdict(set)
        node_id_to_gpu_ids = defaultdict(set)

        for worker_id, (node_id, gpu_ids) in enumerate(node_ids_and_gpu_ids):
            node_id_to_worker_id[node_id].add(worker_id)
            node_id_to_gpu_ids[node_id].update(gpu_ids)

        futures = []
        for node_id, gpu_ids in node_id_to_gpu_ids.items():
            gpu_ids = sorted(gpu_ids)
            all_gpu_ids = ",".join(gpu_ids)

            def set_gpu_ids():
                os.environ["CUDA_VISIBLE_DEVICES"] = all_gpu_ids

            for worker_id in node_id_to_worker_id[node_id]:
                futures.append(
                    self.worker_group.execute_single_async(worker_id, set_gpu_ids)
                )
        ray.get(futures)

    def _create_rank_world_size_mappings(self) -> List[Dict]:
        """Create rank and world size mappings for workers.
        There are three maps returned:
            - local_rank_map, which maps from worker world_rank to local_rank.
            - local_world_size_map, which maps from world_rank to local_world_size
            - node_rank_map, which maps from world rank to node rank

        Example:
            Worker 0: 0.0.0.0
            Worker 1: 0.0.0.0
            Worker 2: 0.0.0.1
            Worker 3: 0.0.0.0
            Worker 4: 0.0.0.1

            Workers 0, 1, 3 are on 0.0.0.0.
            Workers 2, 4 are on 0.0.0.1.

            Expected local_rank_map:
            {
                0 -> 0,
                1 -> 1,
                2 -> 0,
                3 -> 2,
                4 -> 1
            }

            Expected local_world_size_map:
            {
                0 -> 3,
                1 -> 3,
                2 -> 2,
                3 -> 3,
                4 -> 2
            }

            Expected node_rank_map:
            {
                0 -> 0,
                1 -> 0,
                2 -> 1,
                3 -> 0,
                4 -> 1
            }

        """
        local_rank_map = {}  # map from world rank to local rank
        local_world_size_map = {}  # map from world rank to local world size
        node_rank_map = {}  # map from world rank to node rank
        node_ips = {}  # map from node ip to node index
        node_cnt = 0  # count the number of nodes

        ip_dict = defaultdict(int)  # map from node ip to the number of workers on it.
        for world_rank in range(len(self.worker_group)):
            worker = self.worker_group.workers[world_rank]
            node_ip = worker.metadata.node_ip
            local_rank_map[world_rank] = ip_dict[node_ip]
            ip_dict[node_ip] += 1

            if node_ip not in node_ips:
                node_ips[node_ip] = node_cnt
                node_cnt += 1
            node_rank_map[world_rank] = node_ips[node_ip]

        for world_rank in range(len(self.worker_group)):
            worker = self.worker_group.workers[world_rank]
            node_ip = worker.metadata.node_ip
            local_world_size_map[world_rank] = ip_dict[node_ip]

        return local_rank_map, local_world_size_map, node_rank_map

    def start_training(
        self,
        train_func: Callable[[], T],
        datasets: Dict[str, Dataset],
        data_config: DataConfig,
        checkpoint: Optional[Checkpoint] = None,
        on_session_init: Callable[[], None] = None,
    ) -> None:
        """Executes a training function on all workers in a separate thread.

        ``finish_training`` should be called after this.

        Args:
            train_func: The training function to run on each worker.
            datasets: The base datasets.
            data_config: The config object for creating dataset shards for workers.
            checkpoint: The checkpoint data that
                should be loaded onto each worker and accessed by the
                training function via ``session.get_checkpoint()``. If this
                is ``None`` then no checkpoint will be loaded.
        """
        use_detailed_autofilled_metrics = env_integer(
            ENABLE_DETAILED_AUTOFILLED_METRICS_ENV, 0
        )
        use_lazy_checkpointing = not env_integer(DISABLE_LAZY_CHECKPOINTING_ENV, 0)

        # First initialize the session.
        def initialize_session(
            train_func,
            world_rank,
            local_rank,
            node_rank,
            local_world_size,
            world_size,
            trial_info,
            checkpoint,
            dataset_shard,
            encode_data_fn,
            checkpoint_keep_all_ranks,
            checkpoint_upload_from_workers,
            storage,
        ):
            try:
                init_session(
                    training_func=train_func,
                    world_rank=world_rank,
                    local_rank=local_rank,
                    node_rank=node_rank,
                    local_world_size=local_world_size,
                    world_size=world_size,
                    trial_info=trial_info,
                    dataset_shard=dataset_shard,
                    checkpoint=checkpoint,
                    encode_data_fn=encode_data_fn,
                    detailed_autofilled_metrics=use_detailed_autofilled_metrics,
                    enable_lazy_checkpointing=use_lazy_checkpointing,
                    checkpoint_keep_all_ranks=checkpoint_keep_all_ranks,
                    checkpoint_upload_from_workers=(checkpoint_upload_from_workers),
                    storage=storage,
                )
            except ValueError:
                raise TrainBackendError(
                    "Attempting to start training but a "
                    "previous training run is still ongoing. "
                    "You must call `finish_training` before "
                    "calling `start_training` again."
                )

        if self.dataset_shards is None:
            actors = [worker.actor for worker in self.worker_group.workers]
            node_ids = [worker.metadata.node_id for worker in self.worker_group.workers]
            self.dataset_shards = data_config.configure(
                datasets,
                world_size=len(self.worker_group),
                worker_handles=actors,
                worker_node_ids=node_ids,
            )

        (
            local_rank_map,
            local_world_size_map,
            node_rank_map,
        ) = self._create_rank_world_size_mappings()

        futures = []
        for index in range(len(self.worker_group)):
            futures.append(
                self.worker_group.execute_single_async(
                    index,
                    initialize_session,
                    world_rank=index,
                    local_rank=local_rank_map[index],
                    node_rank=node_rank_map[index],
                    local_world_size=local_world_size_map[index],
                    world_size=len(self.worker_group),
                    trial_info=self._trial_info,
                    train_func=train_func,
                    dataset_shard=self.dataset_shards[index],
                    checkpoint=checkpoint,
                    encode_data_fn=self._backend._encode_data,
                    checkpoint_keep_all_ranks=self._checkpoint_keep_all_ranks,
                    checkpoint_upload_from_workers=(
                        self._checkpoint_upload_from_workers
                    ),
                    # Pass the Trainable's shared storage context to the Train workers
                    storage=get_storage_context() if _use_storage_context() else None,
                )
            )

        self._backend.on_training_start(self.worker_group, self._backend_config)

        self.get_with_failure_handling(futures)

        if on_session_init:
            on_session_init()

        # Run the training function asynchronously in its own thread.
        def train_async():
            session = get_session()
            session.start()

        self.worker_group.execute_async(train_async)

    def get_next_results(self) -> Optional[List[TrainingResult]]:
        """Fetches the next ``TrainingResult`` from each worker.

        Each ``TrainingResult`` is expected to correspond to the same step from
        each worker (e.g. the same call to ``session.report()``).

        Returns:
            A list of ``TrainingResult``s with the same
            ``TrainingResultType``, or ``None`` if there are no more results.
        """

        def get_next():
            session = _get_session("get_next_results")
            try:
                result = session.get_next()
            except RuntimeError:
                # Training thread has not been started yet.
                raise TrainBackendError(
                    "`get_next_results` has been called "
                    "before `start_training`. Please call "
                    "`start_training` before "
                    "`get_next_results`."
                )

            return result

        # Get next result from each worker.
        futures = self.worker_group.execute_async(get_next)
        results = self.get_with_failure_handling(futures)

        # Check if any worker returned None.
        if any(r is None for r in results):
            # Either all workers have results or none of them do.
            if not all(r is None for r in results):
                raise RuntimeError(
                    "Some workers returned results while "
                    "others didn't. Make sure that "
                    "`session.report()` are called the "
                    "same number of times on all workers."
                )
            else:
                # Return None if all results are None.
                return None
        first_result = results[0]
        result_type = first_result.type
        if any(r.type != result_type for r in results):
            raise RuntimeError(
                "Some workers returned results with "
                "different types. Make sure that "
                "`session.report()` are called the "
                "same number of times on all workers."
            )

        return results

    def _set_checkpoint_uri(self, uri: str):
        """Tell remote sessions where to upload the chekcpoint."""

        def set_uri():
            session = _get_session("_set_checkpoint_uri")
            session._set_checkpoint_uri(uri)

        futures = self.worker_group.execute_async(set_uri)
        self.get_with_failure_handling(futures)

    def pause_reporting(self):
        """Disable workers from enqueuing results from ``session.report()``.

        Note: Already reported results may still be enqueued at this point,
              and should be handled appropriately.
        """

        def pause_session_reporting():
            session = _get_session("pause_reporting")
            return session.pause_reporting()

        futures = self.worker_group.execute_async(pause_session_reporting)
        self.get_with_failure_handling(futures)

    def finish_training(self):
        """Finish training and return final results. Propagate any exceptions.

        Blocks until training is finished on all workers.

        Assumes `start_training` has already been called.

        Returns:
            A list of return values from calling ``train_func`` on each worker.
                Each item corresponds to the return value from a single worker.
        """

        def end_training():
            session = _get_session("finish_training")
            try:
                # session.finish raises any Exceptions from training.
                output = session.finish()
            finally:
                # Shutdown session even if session.finish() raises an
                # Exception.
                shutdown_session()

            return output

        futures = self.worker_group.execute_async(end_training)
        results = self.get_with_failure_handling(futures)
        return results

    def get_with_failure_handling(self, remote_values):
        """Gets the remote values while handling for worker failures.

        This method should be called instead of ``ray.get()`` directly in
        order to handle worker failures.

        If a worker failure is identified, backend specific failure handling
        is executed and a ``TrainingWorkerError`` is raised.

        Args:
            remote_values: List of object refs representing functions
                that may fail in the middle of execution. For example, running
                a Train training loop in multiple parallel actor calls.
        Returns:
            The resolved objects represented by the passed in ObjectRefs.
        """
        success, exception = check_for_failure(remote_values)
        if success:
            return ray.get(remote_values)
        else:
            self._last_failure = exception
            self._increment_failures()
            logger.warning(
                "Failure identified during training. Restarting all workers and "
                "continuing training from latest checkpoint."
            )
            self._restart()
            raise TrainingWorkerError

    def shutdown(self, graceful_termination: bool = True):
        """Shuts down the workers in the worker group.

        Args:
            graceful_termination: If set to True, attempt to clean up the backend
                before terminating the Ray actors.

        """
        if graceful_termination:
            try:
                self._backend.on_shutdown(self.worker_group, self._backend_config)
            except RayActorError:
                logger.warning(
                    "Graceful shutdown of backend failed. This is "
                    "expected if one of the workers has crashed."
                )

        if graceful_termination:
            self.worker_group.shutdown()
        else:
            self.worker_group.shutdown(patience_s=0)
        self.worker_group = InactiveWorkerGroup()

        if self._placement_group:
            remove_placement_group(self._placement_group)
            self._placement_group = None

        self.dataset_shards = None

    def is_started(self):
        return not isinstance(self.worker_group, InactiveWorkerGroup)

    def _restart(self):
        self.worker_group.shutdown()
        if self._initialization_hook is not None:
            initialization_hook = self._initialization_hook
        else:
            initialization_hook = None
        if self._placement_group:
            remove_placement_group(self._placement_group)
            self._placement_group = None
        self.start(initialization_hook=initialization_hook)

    def _increment_failures(self):
        self._num_failures += 1
        if self._num_failures >= self._max_failures:
            failure = self._last_failure
            self._last_failure = None
            if self._max_failures > 0:
                exc = RuntimeError(
                    "Training has failed after " f"{self._num_failures} " "attempts."
                )
                raise exc.with_traceback(None) from failure
            else:
                raise failure

    def get_worker_group(self):
        return self.worker_group

    def _get_num_failures(self):
        return self._num_failures


class InactiveWorkerGroupError(Exception):
    """Raised when underlying worker group is inactive."""


class InactiveWorkerGroup:
    # TODO: fix inheritence. perhaps create WorkerGroupInterface.

    # Need to define getstate and setstate so that getattr does not screwup
    # pickling. See https://stackoverflow.com/a/50888571/11249691
    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, name):
        raise InactiveWorkerGroupError()

    def __len__(self):
        raise InactiveWorkerGroupError()


def _get_session(method_name: str):
    # Get the session for this worker.
    session = get_session()
    if not session:
        # Session is not initialized yet.
        raise TrainBackendError(
            f"`{method_name}` has been called "
            "before `start_training`. Please call "
            "`start_training` before "
            f"`{method_name}`."
        )
    return session
