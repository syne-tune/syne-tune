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
from typing import Optional, List

from syne_tune.optimizer.schedulers.hyperband_promotion import (
    PromotionRungEntry,
    PromotionRungSystem,
)
from syne_tune.optimizer.schedulers.hyperband_stopping import (
    Rung,
    StoppingRungSystem,
)

logger = logging.getLogger(__name__)


class RUSHDecider:
    """
    Implements the additional decision logic according to the RUSH algorithm.
    It is used as part of :class:`RUSHStoppingRungSystem` and
    :class:`RUSHPromotionRungSystem`. Reference:

        | A resource-efficient method for repeated HPO and NAS.
        | Giovanni Zappella, David Salinas, Cédric Archambeau.
        | AutoML workshop @ ICML 2021.

    For a more detailed description, refer to
    :class:`~syne_tune.optimizer.schedulers.transfer_learning.RUSHScheduler`.

    :param num_threshold_candidates: Number of threshold candidates
    :param mode: "min" or "max"
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
        self, task_continues: bool, trial_id: str, metric_val: float, resource: int
    ) -> bool:
        if not task_continues:
            return False
        if self._is_in_points_to_evaluate(trial_id):
            self._thresholds[resource] = self._return_better(
                self._thresholds.get(resource), metric_val
            )
            return True
        return self._meets_threshold(metric_val, resource)

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

    def _meets_threshold(self, metric_val: float, resource: int) -> bool:
        return (
            self._return_better(self._thresholds.get(resource), metric_val)
            == metric_val
        )


class RUSHStoppingRungSystem(StoppingRungSystem):
    """
    Implementation for RUSH algorithm, stopping variant.

    Additional arguments on top of base class
    :class:`~syne_tune.optimizer.schedulers.hyperband_stopping.StoppingRungSystem`:

    :param num_threshold_candidates: Number of threshold candidates
    """

    def __init__(
        self,
        rung_levels: List[int],
        promote_quantiles: List[float],
        metric: str,
        mode: str,
        resource_attr: str,
        max_t: int,
        num_threshold_candidates: int,
    ):
        super().__init__(
            rung_levels, promote_quantiles, metric, mode, resource_attr, max_t
        )
        self._decider = RUSHDecider(num_threshold_candidates, mode)

    def _task_continues(
        self,
        trial_id: str,
        metric_val: float,
        rung: Rung,
    ) -> bool:
        task_continues = super()._task_continues(trial_id, metric_val, rung)
        return self._decider.task_continues(
            task_continues, trial_id, metric_val, rung.level
        )


class RUSHPromotionRungSystem(PromotionRungSystem):
    """
    Implementation for RUSH algorithm, promotion variant.

    Additional arguments on top of base class
    :class:`~syne_tune.optimizer.schedulers.hyperband_promotion.PromotionRungSystem`:

    :param num_threshold_candidates: Number of threshold candidates
    """

    def __init__(
        self,
        rung_levels: List[int],
        promote_quantiles: List[float],
        metric: str,
        mode: str,
        resource_attr: str,
        max_t: int,
        num_threshold_candidates: int,
    ):
        super().__init__(
            rung_levels, promote_quantiles, metric, mode, resource_attr, max_t
        )
        self._decider = RUSHDecider(num_threshold_candidates, mode)

    def _is_promotable_trial(self, entry: PromotionRungEntry, resource: int) -> bool:
        task_continues = super()._is_promotable_trial(entry, resource)
        return self._decider.task_continues(
            task_continues, entry.trial_id, entry.metric_val, resource
        )
