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
from typing import Optional

from syne_tune.optimizer.schedulers.hyperband_stopping import (
    quantile_cutoff,
    RungSystem,
)


class PromotionRungSystem(RungSystem):
    """
    Implements both the promotion and stopping logic for an asynchronous
    variant of Hyperband, known as ASHA:

    https://arxiv.org/abs/1810.05934

    In ASHA, configs sit paused at milestones (rung levels) until they get
    promoted, which means that a free task picks up their evaluation until
    the next milestone.

    The rule to decide whether a paused trial is promoted (or remains
    paused) is the same as in :class:`StoppingRungSystem`, except that
    `continues` becomes `gets_promoted`. If several paused trials in a
    rung can be promoted, the one with the best metric value is chosen.

    Note: Say that an evaluation is resumed from level resume_from. If the
    train_fn does not implement pause & resume, it needs to start training from
    scratch, in which case metrics are reported for every epoch, also those <
    resume_from. At least for some modes of fitting the searcher model to data,
    this would lead to duplicate target values for the same extended config
    (x, r), which we want to avoid. The solution is to maintain resume_from in
    the data for the terminator (see `PromotionRungSystem._running`). Given
    this, we can report in `on_task_report` that the current metric data should
    not be used for the searcher model (`ignore_data = True`), namely as long
    as the evaluation has not yet gone beyond level resume_from.
    """

    def __init__(
        self, rung_levels, promote_quantiles, metric, mode, resource_attr, max_t
    ):
        super().__init__(rung_levels, promote_quantiles, metric, mode, resource_attr)
        # The data entry in `_rungs` is a dict mapping trial_id to
        # (metric_value, was_promoted)
        self.max_t = max_t
        # `_running` maps `trial_id `to `dict(milestone, resume_from)`.
        # The tasks runs trial `trial_id` until resource reaches milestone.
        # The `resume_from` field can be None. If not, the task is running a
        # trial which has been resumed from rung level `resume_from.` This info
        # is required for `on_task_report` to properly report `ignore_data`.
        self._running = dict()

    def _cutoff(self, recorded: dict, prom_quant: float):
        values = [x[0] for x in recorded.values()]
        return quantile_cutoff(values, prom_quant, self._mode)

    def _find_promotable_trial(
        self, recorded: dict, prom_quant: float, resource: int
    ) -> Optional[str]:
        """
        Check whether any not yet promoted entry in `recorded` is
        promotable, i.e. its value is better or equal to the cutoff
        based on `prom_quant`. If there are several such, the one with the
        best value is chosen.

        :param recorded: Dict to scan
        :param prom_quant: Quantile for promotion
        :param resource: Resource level of rung (i.e., amount of resource
            spent by trials at this rung)
        :return: trial_id if found, otherwise None
        """
        ret_id = None
        # Code is written for 'max' mode. For 'min', we just negate all
        # criterion values
        sign = 1 - 2 * (self._mode == "min")
        cutoff = self._cutoff(recorded, prom_quant)
        if cutoff is not None:
            # Best id among trials paused at this rung (i.e., not yet promoted)
            trial_id, val = max(
                (
                    (k, v[0])
                    for k, v in recorded.items()
                    if self._is_promotable_trial(k, v[0], not v[1], resource)
                ),
                key=lambda x: sign * x[1],
                default=(None, 0.0),
            )
            if trial_id is not None and sign * (val - cutoff) >= 0:
                ret_id = trial_id
        return ret_id

    def _is_promotable_trial(
        self, trial_id: str, metric_value: float, is_paused: bool, resource: int
    ) -> bool:
        """
        Checks whether trial in rung level is promotable in principle, used
        as filter in `_find_promotable_trial`. Can be used in subclasses to
        sharpen the condition for promotion.

        """
        return is_paused

    @staticmethod
    def _mark_as_promoted(recorded: dict, trial_id: str):
        curr_val = recorded[trial_id]
        assert not curr_val[-1]  # Sanity check
        recorded[trial_id] = curr_val[:-1] + (True,)

    def _effective_max_t(self):
        """
        The following method is used in on_task_schedule to control the maximum
        amount of resources allocated to a single configuration during the
        optimization. For ASHA it's just a constant value.
        """
        return self.max_t

    def on_task_schedule(self) -> dict:
        """
        Used to implement _promote_trial of scheduler. Searches through rungs
        to find a trial which can be promoted. If one is found, we return the
        trial_id and other info (current milestone, milestone to be promoted
        to). We also mark the trial as being promoted at the rung level it sits
        right now.
        """
        trial_id = None
        next_milestone = self.max_t
        milestone = None
        recorded = None
        for rung in self._rungs:
            _milestone = rung.level
            prom_quant = rung.prom_quant
            _recorded = rung.data
            if _milestone < self._effective_max_t():
                trial_id = self._find_promotable_trial(
                    _recorded, prom_quant, rung.level
                )
            if trial_id is not None:
                recorded = _recorded
                milestone = _milestone
                break
            next_milestone = _milestone
        ret_dict = dict()
        if trial_id is not None:
            self._mark_as_promoted(recorded, trial_id)
            ret_dict = {
                "trial_id": trial_id,
                "resume_from": milestone,
                "milestone": next_milestone,
            }
        return ret_dict

    def on_task_add(self, trial_id: str, skip_rungs: int, **kwargs):
        """
        Called when new task is started. Depending on kwargs['new_config'],
        this could start an evaluation (True) or promote an existing config
        to the next milestone (False). In the latter case, kwargs contains
        additional information about the promotion.
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
        self, trial_id: str, result: dict, recorded: dict
    ):
        metric_value = result[self._metric]
        assert trial_id not in recorded  # Sanity check
        recorded[trial_id] = (metric_value, False)

    def on_task_report(self, trial_id: str, result: dict, skip_rungs: int) -> dict:
        """
        Decision on whether task may continue (task_continues = True), or
        should be paused (task_continues = False).
        milestone_reached is a flag whether resource coincides with a
        milestone.
        For this scheduler, we have that

            task_continues == not milestone_reached,

        since a trial is always paused at a milestone.

        `ignore_data` is True if a result is received from a resumed trial
        at a level <= `resume_from`. This happens if checkpointing is not
        implemented (or not used), because resumed trials are started from
        scratch then. These metric values should in general be ignored.

        :param trial_id:
        :param result: Reported metrics
        :param skip_rungs: This number of lowest rung levels are not
            considered milestones for this task
        :return: dict(task_continues, milestone_reached, next_milestone,
                      ignore_data)
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
            try:
                rung_pos = next(
                    i for i, v in enumerate(self._rungs) if v.level == milestone
                )
                # Register metric_value at rung level (as not promoted)
                recorded = self._rungs[rung_pos].data
                self._register_metrics_at_rung_level(trial_id, result, recorded)
                next_milestone = (
                    self._rungs[rung_pos - 1].level if rung_pos > 0 else self.max_t
                )
            except StopIteration:
                # milestone not a rung level. This can happen, in particular
                # if milestone == self.max_t
                pass
        return {
            "task_continues": not milestone_reached,
            "milestone_reached": milestone_reached,
            "next_milestone": next_milestone,
            "ignore_data": ignore_data,
        }

    def on_task_remove(self, trial_id: str):
        del self._running[trial_id]
