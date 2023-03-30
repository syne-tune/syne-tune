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
from typing import List, Optional, Tuple, Dict, Any
import logging

from syne_tune.optimizer.schedulers.hyperband_stopping import Rung
from syne_tune.optimizer.schedulers.hyperband_promotion import (
    PromotionRungEntry,
    PromotionRungSystem,
)

logger = logging.getLogger(__name__)


class CostPromotionRungEntry(PromotionRungEntry):
    """
    Appends ``cost_val`` to the superclass. This is the cost value
    :math:`c(x, r)` recorded for the trial at the resource level.
    """

    def __init__(
        self,
        trial_id: str,
        metric_val: float,
        cost_val: float,
        was_promoted: bool = False,
    ):
        super().__init__(trial_id, metric_val, was_promoted)
        self.cost_val = cost_val


class CostPromotionRungSystem(PromotionRungSystem):
    """
    Cost-aware extension of promotion-based asynchronous Hyperband (ASHA).

    This code is equivalent with base
    :class:`~syne_tune.optimizer.schedulers.hyperband_promotion.PromotionRungSystem`,
    except the "promotable" condition in :meth:`_find_promotable_trial` is
    replaced.

    When a config :math:`\mathbf{x}` reaches rung level :math:`r`, the result
    includes a metric :math:`f(\mathbf{x}, r)`, but also a cost
    :math:`c(\mathbf{x}, r)`. The latter is the cost (e.g., training time) spent
    to reach level :math:`r`.

    Consider all trials who reached rung level :math:`r` (whether promoted from
    there or still paused there), ordered w.r.t. :math:`f(\mathbf{x}, r)`, best
    first, and let their number be :math:`N`. Define

    .. math::

       C(r, k) = \sum_{i\le k} c(\mathbf{x}_i, r)

    For a promotion quantile :math:`q`, define

    .. math::

        K = \max_k \mathrm{I}[ C(r, k) \le q  C(r, N) ]

    Any trial not yet promoted and ranked :math:`\le K` can be promoted.
    As usual, we scan rungs from the top. If several trials are promotable,
    the one with the best metric value is promoted.

    Note that costs :math:`c(\mathbf{x}, r)` reported via ``cost_attr`` need to
    be total costs of a trial. If the trial is paused and resumed, partial costs
    have to be added up. See :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`
    for how this works.
    """

    def __init__(
        self,
        rung_levels: List[int],
        promote_quantiles: List[float],
        metric: str,
        mode: str,
        resource_attr: str,
        cost_attr: str,
        max_t: int,
    ):
        super().__init__(
            rung_levels, promote_quantiles, metric, mode, resource_attr, max_t
        )
        self._cost_attr = cost_attr

    def _find_promotable_trial(self, rung: Rung) -> Optional[Tuple[str, int]]:
        """
        The promotability condition depends on the cost values (see header
        comment).
        """
        result = None
        if len(rung) > 1:
            cost_threshold = sum(x.cost_val for x in rung.data) * rung.prom_quant
            sum_costs = 0
            # ``rung.data`` is ordered, with best metric value first
            for pos, entry in enumerate(rung.data):
                sum_costs += entry.cost_val
                if sum_costs > cost_threshold:
                    break  # Nothing to promote
                if self._is_promotable_trial(entry, rung.level):
                    result = (entry.trial_id, pos)
                    break
        return result

    def _register_metrics_at_rung_level(
        self, trial_id: str, result: Dict[str, Any], rung: Rung
    ):
        assert trial_id not in rung  # Sanity check
        rung.add(
            CostPromotionRungEntry(
                trial_id=trial_id,
                metric_val=result[self._metric],
                cost_val=result[self._cost_attr],
            )
        )
