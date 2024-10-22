from typing import List, Optional, Callable

from syne_tune.tuner_callback import TunerCallback
from syne_tune.tuning_status import TuningStatus
from syne_tune.optimizer.schedulers.remove_checkpoints import (
    RemoveCheckpointsSchedulerMixin,
)


class RemoveCheckpointsCallback(TunerCallback):
    """
    This implements early removal of checkpoints of paused trials. In order
    for this to work, the scheduler needs to implement
    :meth:`~syne_tune.optimizer.scheduler.TrialScheduler.trials_checkpoints_can_be_removed`.
    """

    def __init__(self):
        self._tuner = None

    def on_tuning_start(self, tuner):
        assert isinstance(
            tuner.scheduler, DefaultRemoveCheckpointsSchedulerMixin
        ), "tuner.scheduler must be of type DefaultRemoveCheckpointsSchedulerMixin"
        self._tuner = tuner

    def on_loop_end(self):
        for trial_id in self._tuner.scheduler.trials_checkpoints_can_be_removed():
            self._tuner.trial_backend.delete_checkpoint(trial_id)


class DefaultRemoveCheckpointsSchedulerMixin(RemoveCheckpointsSchedulerMixin):
    """
    Implements general case of
    :class:`~syne_tune.optimizer.schedulers.remove_checkpoints.RemoveCheckpointsSchedulerMixin`,
    where the callback is of type :class:`RemoveCheckpointsCallback`. This means
    scheduler has to implement :meth:`trials_checkpoints_can_be_removed`.
    """

    def trials_checkpoints_can_be_removed(self) -> List[int]:
        """
        Supports the general case (see header comment).
        This method returns IDs of paused trials for which checkpoints can safely
        be removed. These trials either cannot be resumed anymore, or it is very
        unlikely they will be resumed. Any trial ID needs to be returned only once,
        not over and over. If a trial gets stopped (by returning
        :const:`SchedulerDecision.STOP` in :meth:`on_trial_result`), its checkpoint
        is removed anyway, so its ID does not have to be returned here.

        :return: IDs of paused trials for which checkpoints can be removed
        """
        return []  # Safe default

    def callback_for_checkpoint_removal(
        self, stop_criterion: Callable[[TuningStatus], bool]
    ) -> Optional[TunerCallback]:
        return RemoveCheckpointsCallback()
