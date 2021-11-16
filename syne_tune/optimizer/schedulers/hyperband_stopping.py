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
import numpy as np
from typing import Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RungEntry:
    level: int  # Rung level r_j
    prom_quant: float  # Promotion quantile q_j
    data: Dict  # Data of all previous jobs reaching the level


def quantile_cutoff(values, prom_quant, mode):
    if len(values) < 2:
        # Cannot determine cutoff from one value
        return None
    q = prom_quant if mode == 'min' else (1 - prom_quant)
    return np.quantile(values, q)


class StoppingRungSystem(object):
    """
    Implements stopping rule resembling the median rule. Once a config is
    stopped, it cannot be promoted later on.
    This is different to what has been published as ASHA (see
    :class:`PromotionRungSystem`).
    """
    def __init__(
            self, rung_levels, promote_quantiles, metric, mode, resource_attr):
        # The data entry in _rungs is a dict mapping task_key to
        # reward_value
        assert len(rung_levels) == len(promote_quantiles)
        self._mode = mode
        self._metric = metric
        self._resource_attr = resource_attr
        self._rungs = [
            RungEntry(level=x, prom_quant=y, data=dict())
            for x, y in reversed(list(zip(rung_levels, promote_quantiles)))]

    def on_task_schedule(self):
        return dict()

    def on_task_add(self, trial_id, skip_rungs, **kwargs):
        pass

    def _cutoff(self, recorded, prom_quant):
        values = list(recorded.values())
        return quantile_cutoff(values, prom_quant, self._mode)

    def on_task_report(self, trial_id, result, skip_rungs):
        """
        Decision on whether task may continue (task_continues = True), or
        should be stopped (task_continues = False).
        milestone_reached is a flag whether resource coincides with a
        milestone. If True, next_milestone is the next milestone after
        resource, or None if there is none.

        :param trial_id:
        :param result:
        :param skip_rungs: This number of lowest rung levels are not
            considered milestones for this task
        :return: dict(task_continues, milestone_reached, next_milestone)
        """
        resource = result[self._resource_attr]
        metric_value = result[self._metric]
        task_continues = True
        milestone_reached = False
        next_milestone = None
        if skip_rungs > 0:
            milestone_rungs = self._rungs[:(-skip_rungs)]
        else:
            milestone_rungs = self._rungs
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
                        f"Continueing, but milestone {milestone} has been skipped.")
                else:
                    milestone_reached = True
                    # Enter new metric value before checking condition
                    recorded[trial_id] = metric_value
                    cutoff = self._cutoff(recorded, prom_quant)
                    if cutoff is not None:
                        if self._mode == 'min':
                            task_continues = (metric_value <= cutoff)
                        else:
                            task_continues = (metric_value >= cutoff)
                break
            next_milestone = milestone
        return {
            'task_continues': task_continues,
            'milestone_reached': milestone_reached,
            'next_milestone': next_milestone}

    def on_task_remove(self, trial_id):
        pass

    def get_first_milestone(self, skip_rungs):
        return self._rungs[-(skip_rungs + 1)].level

    def get_milestones(self, skip_rungs):
        if skip_rungs > 0:
            milestone_rungs = self._rungs[:(-skip_rungs)]
        else:
            milestone_rungs = self._rungs
        return [x.level for x in milestone_rungs]

    def snapshot_rungs(self, skip_rungs):
        if skip_rungs > 0:
            _rungs = self._rungs[:(-skip_rungs)]
        else:
            _rungs = self._rungs
        return [(x.level, x.data) for x in _rungs]
