# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from typing import Optional, List, Dict, Any, Tuple

from syne_tune.optimizer.schedulers.hyperband_stopping import (
    RungEntry,
    Rung,
    RungSystem,
    PausedTrialsResult,
)


class PromotionRungEntry(RungEntry):
    """
    Appends ``was_promoted`` to the superclass. This is ``True`` iff the trial
    has been promoted from this rung. Otherwise, the trial is paused at this rung.
    """

    def __init__(self, trial_id: str, metric_val: float, was_promoted: bool = False):
        super().__init__(trial_id, metric_val)
        self.was_promoted = was_promoted


class PromotionRungSystem(RungSystem):
    """
    Implements the promotion logic for an asynchronous variant of Hyperband,
    known as ASHA:

        | Li etal
        | A System for Massively Parallel Hyperparameter Tuning
        | https://arxiv.org/abs/1810.05934

    In ASHA, configs sit paused at milestones (rung levels) until they get
    promoted, which means that a free task picks up their evaluation until
    the next milestone.

    The rule to decide whether a paused trial is promoted (or remains
    paused) is the same as in
    :class:`~syne_tune.optimizer.schedulers.hyperband_stopping.StoppingRungSystem`,
    except that *continues* becomes *gets promoted*. If several paused trials
    in a rung can be promoted, the one with the best metric value is chosen.

    Note: Say that an evaluation is resumed from level ``resume_from``. If the
    trial evaluation function does not implement pause & resume, it needs to
    start training from scratch, in which case metrics are reported for every
    epoch, also those ``< resume_from``. At least for some modes of fitting the
    searcher model to data, this would lead to duplicate target values for the
    same extended config :math:`(x, r)`, which we want to avoid. The solution is to
    maintain ``resume_from`` in the data for the terminator (see :attr:`_running`).
    Given this, we can report in :meth:`on_task_report` that the current metric
    data should not be used for the searcher model (``ignore_data = True``), namely
    as long as the evaluation has not yet gone beyond level ``resume_from``.
    """

    def __init__(
        self,
        rung_levels: List[int],
        promote_quantiles: List[float],
        metric: str,
        mode: str,
        resource_attr: str,
        max_t: int,
    ):
        super().__init__(
            rung_levels, promote_quantiles, metric, mode, resource_attr, max_t
        )
        # The data entry in ``_rungs`` is a dict mapping trial_id to
        # (metric_value, was_promoted)
        # ``_running`` maps ``trial_id ``to ``dict(milestone, resume_from)``.
        # The tasks runs trial ``trial_id`` until resource reaches milestone.
        # The ``resume_from`` field can be None. If not, the task is running a
        # trial which has been resumed from rung level ``resume_from.`` This info
        # is required for ``on_task_report`` to properly report ``ignore_data``.
        self._running = dict()

    def _find_promotable_trial(self, rung: Rung) -> Optional[Tuple[str, int]]:
        """
        Check whether any not yet promoted entry in ``rung`` is
        promotable, i.e. it lies left of (or is equal to) the pivot.

        :param rung: Rung to scan
        :return: ``(trial_id, pos)`` if found, where ``pos`` is the position
            in ``rung.data``, otherwise ``None``
        """
        cutoff = rung.quantile()
        if cutoff is None:
            return None
        # Find best among paused trials (not yet promoted)
        result, metric_val = None, None
        resource = rung.level
        for pos, entry in enumerate(rung.data):
            if self._is_promotable_trial(entry, resource):
                result = (entry.trial_id, pos)
                metric_val = entry.metric_val
                break
        sign = 1 - 2 * (self._mode == "min")
        if result is not None and sign * (metric_val - cutoff) < 0:
            # Best paused trial is not good enough to be promoted
            result = None
        return result

    def _is_promotable_trial(self, entry: PromotionRungEntry, resource: int) -> bool:
        """
        Checks whether trial in rung level is promotable in principle, used
        as filter in :meth:`_find_promotable_trial`. Can be used in subclasses
        to sharpen the condition for promotion.

        :param entry: Entry of rung in question
        :param resource: Rung level
        :return: Should this entry be promoted, given it fulfils the pivot
            condition?
        """
        return not entry.was_promoted

    @staticmethod
    def _mark_as_promoted(rung: Rung, pos: int, trial_id: Optional[int] = None):
        entry = rung.pop(pos)
        assert not entry.was_promoted
        if trial_id is not None:
            assert (
                entry.trial_id == trial_id
            ), f"Entry at position {pos} should have trial_id = {trial_id}, but has trial_id = {entry.trial_id}"
        entry.was_promoted = True
        rung.add(entry)

    def _effective_max_t(self):
        """
        The following method is used in on_task_schedule to control the maximum
        amount of resources allocated to a single configuration during the
        optimization. For ASHA it's just a constant value.
        """
        return self._max_t

    def on_task_schedule(self, new_trial_id: str) -> Dict[str, Any]:
        """
        Used to implement
        :meth:`~syne_tune.optimizer.schedulers.HyperbandScheduler._promote_trial`.
        Searches through rungs to find a trial which can be promoted. If one is
        found, we return the ``trial_id`` and other info (current milestone,
        milestone to be promoted to). We also mark the trial as being promoted
        at the rung level it sits right now.
        """
        trial_id, pos = None, None
        next_milestone = self._max_t
        milestone = None
        rung = None
        for _rung in self._rungs:
            _milestone = _rung.level
            if _milestone < self._effective_max_t():
                result = self._find_promotable_trial(_rung)
                if result is not None:
                    rung = _rung
                    milestone = _milestone
                    trial_id, pos = result
                    break
            next_milestone = _milestone
        ret_dict = dict()
        if trial_id is not None:
            self._mark_as_promoted(rung, pos)
            ret_dict = {
                "trial_id": trial_id,
                "resume_from": milestone,
                "milestone": next_milestone,
            }
        return ret_dict

    def on_task_add(self, trial_id: str, skip_rungs: int, **kwargs):
        """
        Called when new task is started. Depending on ``kwargs["new_config"]``,
        this could start an evaluation (``True``) or promote an existing config
        to the next milestone (``False``). In the latter case, ``kwargs`` contains
        additional information about the promotion (in "milestone",
        "resume_from").

        :param trial_id: ID of trial to be started
        :param skip_rungs: This number of the smallest rung levels are not
            considered milestones for this task
        :param kwargs: Additional arguments
        """
        new_config = kwargs.get("new_config", True)
        if new_config:
            # New trial
            milestone = self.get_first_milestone(skip_rungs)
            resume_from = None
        else:
            # Existing trial is resumed
            # Note that self._rungs has already been updated in
            # on_task_schedule
            milestone = kwargs["milestone"]
            resume_from = kwargs["resume_from"]
            assert resume_from < milestone  # Sanity check
        self._running[trial_id] = {"milestone": milestone, "resume_from": resume_from}

    def _register_metrics_at_rung_level(
        self, trial_id: str, result: Dict[str, Any], rung: Rung
    ):
        assert trial_id not in rung  # Sanity check
        rung.add(PromotionRungEntry(trial_id=trial_id, metric_val=result[self._metric]))

    def on_task_report(
        self, trial_id: str, result: Dict[str, Any], skip_rungs: int
    ) -> Dict[str, Any]:
        """
        Decision on whether task may continue (``task_continues=True``), or
        should be paused (``task_continues=False``).
        ``milestone_reached`` is a flag whether resource coincides with a
        milestone.
        For this scheduler, we have that

            ``task_continues == not milestone_reached``,

        since a trial is always paused at a milestone.

        ``ignore_data`` is True if a result is received from a resumed trial
        at a level ``<= resume_from``. This happens if checkpointing is not
        implemented (or not used), because resumed trials are started from
        scratch then. These metric values should in general be ignored.

        :param trial_id: ID of trial which reported results
        :param result: Reported metrics
        :param skip_rungs: This number of smallest rung levels are not
            considered milestones for this task
        :return: ``dict(task_continues, milestone_reached, next_milestone,
            ignore_data)``
        """
        resource = result[self._resource_attr]
        milestone_reached = False
        next_milestone = None
        milestone = self._running[trial_id]["milestone"]
        resume_from = self._running[trial_id]["resume_from"]
        ignore_data = (resume_from is not None) and (resource <= resume_from)
        if resource >= milestone:
            assert resource == milestone, (
                f"trial_id {trial_id}: resource = {resource} > {milestone} "
                + "milestone. Make sure to report time attributes covering "
                + "all milestones"
            )
            milestone_reached = True
            rung_pos = self._rung_pos_for_level(milestone)
            if rung_pos is not None:
                # Register metric_value at rung level (as not promoted)
                rung = self._rungs[rung_pos]
                self._register_metrics_at_rung_level(trial_id, result, rung)
                next_milestone = (
                    self._rungs[rung_pos - 1].level if rung_pos > 0 else self._max_t
                )
        return {
            "task_continues": not milestone_reached,
            "milestone_reached": milestone_reached,
            "next_milestone": next_milestone,
            "ignore_data": ignore_data,
        }

    def on_task_remove(self, trial_id: str):
        if trial_id in self._running:
            del self._running[trial_id]

    @staticmethod
    def does_pause_resume() -> bool:
        return True

    def support_early_checkpoint_removal(self) -> bool:
        return True

    def paused_trials(self, resource: Optional[int] = None) -> PausedTrialsResult:
        result = []
        if resource is None:
            rungs = self._rungs
        else:
            rung_pos = self._rung_pos_for_level(resource)
            rungs = [] if rung_pos is None else [self._rungs[rung_pos]]
        for rung in rungs:
            for pos, entry in enumerate(rung.data):
                if not entry.was_promoted:
                    result.append((entry.trial_id, pos, entry.metric_val, rung.level))
        return result
