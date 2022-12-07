import glob
import logging
import os
from typing import Dict, Optional

from ray.tune.execution import trial_runner
from ray.tune.result import DEFAULT_METRIC
from ray.tune.experiment import Trial
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.debug import log_once

logger = logging.getLogger(__name__)


@DeveloperAPI
class TrialScheduler:
    """Interface for implementing a Trial Scheduler class."""

    CONTINUE = "CONTINUE"  #: Status for continuing trial execution
    PAUSE = "PAUSE"  #: Status for pausing trial execution
    STOP = "STOP"  #: Status for stopping trial execution
    # Caution: Temporary and anti-pattern! This means Scheduler calls
    # into Executor directly without going through TrialRunner.
    # TODO(xwjiang): Deprecate this after we control the interaction
    #  between schedulers and executor.
    NOOP = "NOOP"

    CKPT_FILE_TMPL = "scheduler-state-{}.json"

    _metric = None

    _supports_buffered_results = True

    @property
    def metric(self):
        return self._metric

    @property
    def supports_buffered_results(self):
        return self._supports_buffered_results

    def set_search_properties(
        self, metric: Optional[str], mode: Optional[str], **spec
    ) -> bool:
        """Pass search properties to scheduler.

        This method acts as an alternative to instantiating schedulers
        that react to metrics with their own `metric` and `mode` parameters.

        Args:
            metric: Metric to optimize
            mode: One of ["min", "max"]. Direction to optimize.
            **spec: Any kwargs for forward compatiblity.
                Info like Experiment.PUBLIC_KEYS is provided through here.
        """
        if self._metric and metric:
            return False
        if metric:
            self._metric = metric

        if self._metric is None:
            # Per default, use anonymous metric
            self._metric = DEFAULT_METRIC

        return True

    def on_trial_add(self, trial_runner: "trial_runner.TrialRunner", trial: Trial):
        """Called when a new trial is added to the trial runner."""

        raise NotImplementedError

    def on_trial_error(self, trial_runner: "trial_runner.TrialRunner", trial: Trial):
        """Notification for the error of trial.

        This will only be called when the trial is in the RUNNING state."""

        raise NotImplementedError

    def on_trial_result(
        self, trial_runner: "trial_runner.TrialRunner", trial: Trial, result: Dict
    ) -> str:
        """Called on each intermediate result returned by a trial.

        At this point, the trial scheduler can make a decision by returning
        one of CONTINUE, PAUSE, and STOP. This will only be called when the
        trial is in the RUNNING state."""

        raise NotImplementedError

    def on_trial_complete(
        self, trial_runner: "trial_runner.TrialRunner", trial: Trial, result: Dict
    ):
        """Notification for the completion of trial.

        This will only be called when the trial is in the RUNNING state and
        either completes naturally or by manual termination."""

        raise NotImplementedError

    def on_trial_remove(self, trial_runner: "trial_runner.TrialRunner", trial: Trial):
        """Called to remove trial.

        This is called when the trial is in PAUSED or PENDING state. Otherwise,
        call `on_trial_complete`."""

        raise NotImplementedError

    def choose_trial_to_run(
        self, trial_runner: "trial_runner.TrialRunner"
    ) -> Optional[Trial]:
        """Called to choose a new trial to run.

        This should return one of the trials in trial_runner that is in
        the PENDING or PAUSED state. This function must be idempotent.

        If no trial is ready, return None."""

        raise NotImplementedError

    def debug_string(self) -> str:
        """Returns a human readable message for printing to the console."""

        raise NotImplementedError

    def save(self, checkpoint_path: str):
        """Save trial scheduler to a checkpoint"""
        raise NotImplementedError

    def restore(self, checkpoint_path: str):
        """Restore trial scheduler from checkpoint."""
        raise NotImplementedError

    def has_checkpoint(self, checkpoint_dir: str) -> bool:
        """Should return False if saving/restoring is not implemented."""
        return bool(
            glob.glob(os.path.join(checkpoint_dir, self.CKPT_FILE_TMPL.format("*")))
        )

    def save_to_dir(self, checkpoint_dir: str, session_str: str = "default"):
        """Saves the scheduler state to the checkpoint_dir.

        This is automatically used by Tuner().fit() during a Tune job.

        Args:
            checkpoint_dir: Filepath to experiment dir.
            session_str: Unique identifier of the current run
                session.
        """
        tmp_scheduler_ckpt_path = os.path.join(checkpoint_dir, ".tmp_scheduler_ckpt")
        success = True
        try:
            self.save(tmp_scheduler_ckpt_path)
        except NotImplementedError:
            if log_once("schedulers:save_not_implemented"):
                logger.warning(
                    f"Save is not implemented for {self.__class__.__name__}. "
                    "Skipping scheduler save."
                )
            success = False

        if success and os.path.exists(tmp_scheduler_ckpt_path):
            os.replace(
                tmp_scheduler_ckpt_path,
                os.path.join(checkpoint_dir, self.CKPT_FILE_TMPL.format(session_str)),
            )

    def restore_from_dir(self, checkpoint_dir: str):
        """Restores the state of a scheduler from a given checkpoint_dir.

        Typically, you should use this function to restore from an
        experiment directory such as `~/ray_results/trainable`.
        If there are multiple scheduler checkpoints within the directory,
        this will restore from the most recent one.

        Args:
            checkpoint_dir: Filepath to experiment dir.
        """
        pattern = self.CKPT_FILE_TMPL.format("*")
        full_paths = glob.glob(os.path.join(checkpoint_dir, pattern))
        if not full_paths:
            raise RuntimeError(
                f"Scheduler unable to find checkpoint in {checkpoint_dir}. "
            )
        most_recent_checkpoint = max(full_paths)
        self.restore(most_recent_checkpoint)


@PublicAPI
class FIFOScheduler(TrialScheduler):
    """Simple scheduler that just runs trials in submission order."""

    def on_trial_add(self, trial_runner: "trial_runner.TrialRunner", trial: Trial):
        pass

    def on_trial_error(self, trial_runner: "trial_runner.TrialRunner", trial: Trial):
        pass

    def on_trial_result(
        self, trial_runner: "trial_runner.TrialRunner", trial: Trial, result: Dict
    ) -> str:
        return TrialScheduler.CONTINUE

    def on_trial_complete(
        self, trial_runner: "trial_runner.TrialRunner", trial: Trial, result: Dict
    ):
        pass

    def on_trial_remove(self, trial_runner: "trial_runner.TrialRunner", trial: Trial):
        pass

    def choose_trial_to_run(
        self, trial_runner: "trial_runner.TrialRunner"
    ) -> Optional[Trial]:
        for trial in trial_runner.get_trials():
            if (
                trial.status == Trial.PENDING
                and trial_runner.trial_executor.has_resources_for_trial(trial)
            ):
                return trial
        for trial in trial_runner.get_trials():
            if (
                trial.status == Trial.PAUSED
                and trial_runner.trial_executor.has_resources_for_trial(trial)
            ):
                return trial
        return None

    def debug_string(self) -> str:
        return "Using FIFO scheduling algorithm."
