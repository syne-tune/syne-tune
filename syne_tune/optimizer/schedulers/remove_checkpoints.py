from typing import Optional, Callable

from syne_tune.tuner_callback import TunerCallback
from syne_tune.tuning_status import TuningStatus


class RemoveCheckpointsSchedulerMixin:
    """
    Methods to be implemented by pause-and-resume schedulers (in that
    :meth:`on_trial_result` can return :const:`SchedulerDecision.PAUSE`) which
    support early removal of checkpoints. Typically, model checkpoints are
    retained for paused trials, because they may get resumed later on. This can
    lead to the disk filling up, so removing checkpoints which are no longer
    needed, can be important.

    Early checkpoint removal is implemented as a callback used with
    :class:`~syne_tune.Tuner`, which is created by
    :meth:`callback_for_checkpoint_removal` here.
    """

    def callback_for_checkpoint_removal(
        self, stop_criterion: Callable[[TuningStatus], bool]
    ) -> Optional[TunerCallback]:
        """
        :param stop_criterion: Stopping criterion, as passed to
            :class:`~syne_tune.Tuner`
        :return: CP removal callback, or ``None`` if CP removal is not activated
        """
        raise NotImplementedError
