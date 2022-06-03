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
from typing import Optional

from syne_tune.optimizer.schedulers.hyperband_promotion import PromotionRungSystem
from syne_tune.optimizer.schedulers.hyperband_stopping import StoppingRungSystem

logger = logging.getLogger(__name__)


class RUSHDecider:
    """
    Implements the additional decision logic according to the RUSH algorithm.
    It is used as part of RUSHStoppingRungSystem and RUSHPromotionRungSystem.

    Reference: A resource-efficient method for repeated HPO and NAS.
    Giovanni Zappella, David Salinas, Cédric Archambeau. AutoML workshop @ ICML 2021.

    For a more detailed description, refer to
    :class:`RUSHScheduler`.
    """

    def __init__(self, num_threshold_candidates: int, mode: str):
        if num_threshold_candidates <= 0:
            logger.warning(
                "No threshold candidates provided. 'rush_stopping' will behave exactly like 'stopping'."
            )
        self._num_threshold_candidates = num_threshold_candidates
        self._mode = mode
        self._thresholds = (
            dict()
        )  # thresholds at different resource levels that must be met

    def task_continues(
        self, task_continues: bool, trial_id: str, metric_value: float, resource: int
    ) -> bool:
        if not task_continues:
            return False
        if self._is_in_points_to_evaluate(trial_id):
            self._thresholds[resource] = self._return_better(
                self._thresholds.get(resource), metric_value
            )
            return True
        return self._meets_threshold(metric_value, resource)

    def _is_in_points_to_evaluate(self, trial_id: str) -> bool:
        return int(trial_id) < self._num_threshold_candidates

    def _return_better(self, val1: Optional[float], val2: Optional[float]) -> float:
        if self._mode == "min":
            better_val = min(
                float("inf") if val1 is None else val1,
                float("inf") if val2 is None else val2,
            )
        else:
            better_val = max(
                float("-inf") if val1 is None else val1,
                float("-inf") if val2 is None else val2,
            )
        return better_val

    def _meets_threshold(self, metric_value: float, resource: int) -> bool:
        return (
            self._return_better(self._thresholds.get(resource), metric_value)
            == metric_value
        )


class RUSHStoppingRungSystem(StoppingRungSystem):
    def __init__(self, num_threshold_candidates: int, **kwargs):
        super().__init__(**kwargs)
        self._decider = RUSHDecider(num_threshold_candidates, self._mode)

    def _task_continues(
        self,
        metric_value: float,
        recorded: dict,
        prom_quant: float,
        trial_id: str,
        resource: int,
    ) -> bool:
        task_continues = super()._task_continues(
            metric_value, recorded, prom_quant, trial_id, resource
        )
        return self._decider.task_continues(
            task_continues, trial_id, metric_value, resource
        )


class RUSHPromotionRungSystem(PromotionRungSystem):
    def __init__(self, num_threshold_candidates: int, **kwargs):
        super().__init__(**kwargs)
        self._decider = RUSHDecider(num_threshold_candidates, self._mode)

    def _is_promotable_trial(
        self, trial_id: str, metric_value: float, is_paused: bool, resource: int
    ) -> bool:
        task_continues = super()._is_promotable_trial(
            trial_id, metric_value, is_paused, resource
        )
        return self._decider.task_continues(
            task_continues, trial_id, metric_value, resource
        )
