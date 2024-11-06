import numpy as np
import logging

from dataclasses import dataclass
from typing import Optional, Dict, Any

from syne_tune.backend.trial_status import Trial

logger = logging.getLogger(__name__)


class SchedulerDecision:
    """
    Possible return values of :meth:`TrialScheduler.on_trial_result`, signals the
    tuner how to proceed with the reporting trial.

    The difference between :const:`PAUSE` and :const:`STOP` is important. If a
    trial is stopped, it cannot be resumed afterward. Its checkpoints may be
    deleted. If a trial is paused, it may be resumed in the future, and its
    most recent checkpoint should be retained.
    """

    CONTINUE = "CONTINUE"  #: Status for continuing trial execution
    PAUSE = "PAUSE"  #: Status for pausing trial execution
    STOP = "STOP"  #: Status for stopping trial execution


@dataclass
class TrialSuggestion:
    """Suggestion returned by :meth:`TrialScheduler.suggest`

    :param spawn_new_trial_id: Whether a new ``trial_id`` should be used.
    :param checkpoint_trial_id: Checkpoint of this trial ID should
        be used to resume from. If ``spawn_new_trial_id`` is ``False``, then the
        trial ``checkpoint_trial_id`` is resumed with its previous checkpoint.
    :param config: The configuration which should be evaluated.
    """

    spawn_new_trial_id: bool = True
    checkpoint_trial_id: Optional[int] = None
    config: Optional[dict] = None

    def __post_init__(self):
        if self.spawn_new_trial_id:
            assert (
                self.checkpoint_trial_id is not None or self.config is not None
            ), "Cannot start a new trial without specifying a checkpoint or a config."
        else:
            assert (
                self.checkpoint_trial_id is not None
            ), "A trial-id must be passed to resume a trial."

    @staticmethod
    def start_suggestion(
        config: Dict[str, Any], checkpoint_trial_id: Optional[int] = None
    ) -> "TrialSuggestion":
        """Suggestion to start new trial

        :param config: Configuration to use for the new trial.
        :param checkpoint_trial_id: Use checkpoint of this trial
            when starting the new trial (otherwise, it is started from
            scratch).
        :return: A trial decision that consists in starting a new trial (which
            would receive a new trial-id).
        """
        return TrialSuggestion(
            spawn_new_trial_id=True,
            config=config,
            checkpoint_trial_id=checkpoint_trial_id,
        )

    @staticmethod
    def resume_suggestion(
        trial_id: int, config: Optional[dict] = None
    ) -> "TrialSuggestion":
        """Suggestion to resume a paused trial

        :param trial_id: ID of trial to be resumed (from its checkpoint)
        :param config: Configuration to use for resumed trial
        :return: A trial decision that consists in resuming trial ``trial-id``
            with ``config`` if provided, or the previous configuration used if
            not provided.
        """
        return TrialSuggestion(
            spawn_new_trial_id=False,
            config=config,
            checkpoint_trial_id=trial_id,
        )

    def __str__(self):
        res = f"config {self.config}"
        if self.checkpoint_trial_id is not None:
            res += f" using from trial's checkpoint {self.checkpoint_trial_id}"
        return res


class TrialScheduler:
    """
    Schedulers maintain and drive the logic of an experiment, making decisions
    which configs to evaluate in new trials, and which trials to stop early.

    Some schedulers support pausing and resuming trials. In this case, they
    also drive the decision when to restart a paused trial.

    :param random_seed: Master random seed. Generators used in the
        scheduler or searcher are seeded using :class:`RandomSeedGenerator`.
        If not given, the master random seed is drawn at random here.
    :type random_seed: int, optional
    """

    def __init__(
        self,
        random_seed: int = None,
    ):
        if random_seed is None:
            self.random_seed = np.random.randint(0, 2**31 - 1)
        else:
            self.random_seed = random_seed

    def suggest(self) -> Optional[TrialSuggestion]:
        """Returns a suggestion for a new trial, or one to be resumed

        This method returns ``suggestion`` of type :class:`TrialSuggestion` (unless
        there is no config left to explore, and None is returned).

        If ``suggestion.spawn_new_trial_id`` is ``True``, a new trial is to be
        started with config ``suggestion.config``. Typically, this new trial
        is started from scratch. But if ``suggestion.checkpoint_trial_id`` is
        given, the trial is to be (warm)started from the checkpoint written
        for the trial with this ID. The new trial has ID ``trial_id``.

        If ``suggestion.spawn_new_trial_id`` is ``False``, an existing and currently
        paused trial is to be resumed, whose ID is
        ``suggestion.checkpoint_trial_id``. If this trial has a checkpoint, we
        start from there. In this case, ``suggestion.config`` is optional. If not
        given (default), the config of the resumed trial does not change.
        Otherwise, its config is overwritten by ``suggestion.config`` (see
        :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` with
        ``type="promotion"`` for an example why this can be useful).

        Apart from the HP config, additional fields can be appended to the
        dict, these are passed to the trial function as well.

        :return: Suggestion for a trial to be started or to be resumed, see
            above. If no suggestion can be made, None is returned
        """
        raise NotImplementedError

    def on_trial_add(self, trial: Trial):
        """Called when a new trial is added to the trial runner.

        Additions are normally triggered by ``suggest``.

        :param trial: Trial to be added
        """
        pass

    def on_trial_error(self, trial: Trial):
        pass

    def on_trial_result(self, trial: Trial, result: Dict[str, Any]) -> str:
        """Called on each intermediate result reported by a trial.

        At this point, the trial scheduler can make a decision by returning
        one of :const:`SchedulerDecision.CONTINUE`,
        :const:`SchedulerDecision.PAUSE`, or :const:`SchedulerDecision.STOP`.
        This will only be called when the trial is currently running.

        :param trial: Trial for which results are reported
        :param result: Result dictionary
        :return: Decision what to do with the trial
        """
        return SchedulerDecision.CONTINUE

    def on_trial_complete(self, trial: Trial, result: Dict[str, Any]):
        """Notification for the completion of trial.

        Note that :meth:`on_trial_result` is called with the same result before.
        However, if the scheduler only uses one final report from each
        trial, it may ignore :meth:`on_trial_result` and just use ``result`` here.

        :param trial: Trial which is completing
        :param result: Result dictionary
        """
        pass

    def on_trial_remove(self, trial: Trial):
        """Called to remove trial.

        This is called when the trial is in PAUSED or PENDING state. Otherwise,
        call :meth:`on_trial_complete`.

        :param trial: Trial to be removed
        """
        pass

    def metadata(self) -> Dict[str, Any]:
        """
        :return: Metadata for the scheduler
        """
        return {
            "scheduler_name": str(self.__class__.__name__),
            "scheduler_kwargs": self.__dict__,
        }
