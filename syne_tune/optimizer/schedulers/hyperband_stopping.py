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
import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RungEntry:
    level: int  # Rung level r_j
    prom_quant: float  # Promotion quantile q_j
    data: dict  # Data of all previous jobs reaching the level


def quantile_cutoff(values, prom_quant, mode):
    if len(values) < 2:
        # Cannot determine cutoff from one value
        return None
    q = prom_quant if mode == "min" else (1 - prom_quant)
    return np.quantile(values, q)


class RungSystem:
    """
    Terminology: trials emit results at certain resource levels (e.g.,
    epoch numbers). Some resource levels are rung levels, this is where
    scheduling decisions (stop, continue or pause, resume) are taken.

    For a running trial, the next rung level it will reach is called
    its milestone.

    """

    def __init__(self, rung_levels, promote_quantiles, metric, mode, resource_attr):
        assert len(rung_levels) == len(promote_quantiles)
        self._metric = metric
        self._mode = mode
        self._resource_attr = resource_attr
        # The data entry in `_rungs` is a dict with key trial_id. The
        # value type depends on the subclass, but it contains the
        # metric value
        self._rungs = [
            RungEntry(level=x, prom_quant=y, data=dict())
            for x, y in reversed(list(zip(rung_levels, promote_quantiles)))
        ]

    def on_task_schedule(self) -> dict:
        """
        Called when new task is to be scheduled.
        For a promotion-based rung system, check whether any trial can be
        promoted. If so, return dict with keys `trial_id`, `resume_from`
        (rung level where trial is paused), `milestone` (next rung level
        if will reach, or None).
        If no trial can be promoted, or if the rung system is not
        promotion-based, an empty dict is returned.

        :return: See above
        """
        raise NotImplementedError

    def on_task_add(self, trial_id: str, skip_rungs: int, **kwargs):
        """
        Called when new task is started.

        :param trial_id:
        :param skip_rungs: This number of lowest rung levels are not
            considered milestones for this task
        :param kwargs:
        """
        pass

    def on_task_report(self, trial_id: str, result: dict, skip_rungs: int) -> dict:
        """
        Called when a trial reports metric results. Returns dict with
        `milestone_reached` (trial reaches its milestone), `task_continues`
        (trial should continue; otherwise it is stopped or paused),
        `next_milestone` (next milestone it will reach, or None).
        For certain subclasses, there may be additional fields.

        :param trial_id:
        :param result: Reported metrics
        :param skip_rungs: This number of lowest rung levels are not
            considered milestones for this task
        :return: See above
        """
        raise NotImplementedError

    def on_task_remove(self, trial_id: str):
        """
        Called when task is removed.

        :param trial_id:
        """
        pass

    def get_first_milestone(self, skip_rungs: int) -> int:
        """
        :param skip_rungs: This number of lowest rung levels are not
            considered milestones for this task
        :return: First milestone to be considered
        """
        return self._rungs[-(skip_rungs + 1)].level

    def _milestone_rungs(self, skip_rungs: int) -> List[RungEntry]:
        if skip_rungs > 0:
            return self._rungs[:(-skip_rungs)]
        else:
            return self._rungs

    def get_milestones(self, skip_rungs: int) -> List[int]:
        """
         :param skip_rungs: This number of lowest rung levels are not
            considered milestones for this task
        :return: All milestones to be considered
        """
        milestone_rungs = self._milestone_rungs(skip_rungs)
        return [x.level for x in milestone_rungs]

    def snapshot_rungs(self, skip_rungs: int) -> List[Tuple[int, dict]]:
        """
        A snapshot is a list of rung levels with entries `(level, data)`,
        ordered from top to bottom (highest rung first).

        :param skip_rungs: This number of lowest rung levels are not
            considered milestones for this task
        :return: Snapshot (see above)
        """
        milestone_rungs = self._milestone_rungs(skip_rungs)
        return [(x.level, x.data) for x in milestone_rungs]


class StoppingRungSystem(RungSystem):
    """
    The decision on whether a trial x continues or is stopped at a rung
    level r, is taken in `on_task_report`. To this end, the metric value
    f(x, r) is inserted into r.data. Then:

        continues(x, r)  <==>  f(x, r) <= np.quantile(r.data, r.prom_quant)

    in case `mode == 'min'`. See `_task_continues`.

    """

    def _cutoff(self, recorded, prom_quant):
        values = list(recorded.values())
        return quantile_cutoff(values, prom_quant, self._mode)

    def _task_continues(
        self,
        metric_value: float,
        recorded: dict,
        prom_quant: float,
        trial_id: str,
        resource: int,
    ) -> bool:
        """
        :param metric_value: f(x, r) for trial x at rung r
        :param recorded: Data for rung r (including r(x, r))
        :param prom_quant: Quantile threshold (for mode 'min')
        :param trial_id: ID of trial
        :param resource: Rung level
        :return: Continue trial? Stop otherwise
        """
        cutoff = self._cutoff(recorded, prom_quant)
        if cutoff is None:
            return True
        return metric_value <= cutoff if self._mode == "min" else metric_value >= cutoff

    def on_task_schedule(self) -> dict:
        return dict()

    def on_task_report(self, trial_id: str, result: dict, skip_rungs: int) -> dict:
        resource = result[self._resource_attr]
        metric_value = result[self._metric]
        task_continues = True
        milestone_reached = False
        next_milestone = None
        milestone_rungs = self._milestone_rungs(skip_rungs)
        for rung in milestone_rungs:
            milestone = rung.level
            prom_quant = rung.prom_quant
            recorded = rung.data
            if not (resource < milestone or trial_id in recorded):
                # Note: It is important for model-based searchers that
                # milestones are reached exactly, not jumped over. In
                # particular, if a future milestone is reported via
                # register_pending, its reward value has to be passed
                # later on via update.
                if resource > milestone:
                    logger.warning(
                        f"resource = {resource} > {milestone} = milestone. "
                        "Make sure to report time attributes covering all milestones.\n"
                        f"Continueing, but milestone {milestone} has been skipped."
                    )
                else:
                    milestone_reached = True
                    # Enter new metric value before checking condition
                    recorded[trial_id] = metric_value
                    task_continues = self._task_continues(
                        metric_value=metric_value,
                        recorded=recorded,
                        prom_quant=prom_quant,
                        trial_id=trial_id,
                        resource=resource,
                    )
                break
            next_milestone = milestone
        return {
            "task_continues": task_continues,
            "milestone_reached": milestone_reached,
            "next_milestone": next_milestone,
        }
