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

from syne_tune.optimizer.schedulers.hyperband_promotion import PromotionRungSystem

logger = logging.getLogger(__name__)


class CostPromotionRungSystem(PromotionRungSystem):
    """
    Cost-aware extension of promotion-based asynchronous Hyperband (ASHA).

    This code is equivalent with base :class:`PromotionRungSystem`, except
    the "promotable" condition in `_find_promotable_config` is replaced.

    When a config x reaches rung level r, the result includes a metric
    m(x, r), but also a cost c(x, r). The latter is the cost (e.g., training
    time) spent to reach level r.

    Consider all trials who reached rung level r (whether promoted from there
    or still paused there), ordered w.r.t. m(x, r), best first, and let their
    number be N. Define

        C(r, k) = sum( c(x_i, r) | i <= k)

    For a promotion quantile q, define

        K = max_k [ C(r, k) <= q * C(r, N) ]

    Any trial not yet promoted and ranked <= K can be promoted.

    As usual, we scan rungs from the top. If several trials are promotable,
    the one with the best metric value is promoted.

    Note that costs c(x, r) reported via `cost_attr` need to be total costs of
    a trial. If the trial is paused and resumed, partial costs have to be added
    up. See :class:`HyperbandScheduler` for how this works.

    """

    def __init__(
        self,
        rung_levels,
        promote_quantiles,
        metric,
        mode,
        resource_attr,
        cost_attr,
        max_t,
    ):
        super().__init__(
            rung_levels, promote_quantiles, metric, mode, resource_attr, max_t
        )
        self._cost_attr = cost_attr
        # Note: The data entry in _rungs is now a dict mapping trial_id to
        # (metric_value, cost_value, was_promoted), where metric_value is
        # m(x, r), cost value is c(x, r).

    def _find_promotable_trial(self, recorded, prom_quant, resource):
        """
        Check whether any not yet promoted entry in `recorded` is
        promotable (see header comment). If there are several such, the one
        with the best value is chosen.

        :param recorded: Dict to scan
        :param prom_quant: Quantile for promotion
        :param resource: Amount of resources spent on the rung.
        :return: trial_id if found, otherwise None
        """
        ret_id = None
        if len(recorded) > 1:
            sign = 2 * (self._mode == "min") - 1
            # Sort best-first
            sorted_record = sorted(
                ((k,) + v for k, v in recorded.items()), key=lambda x: x[1] * sign
            )
            cost_threshold = sum(x[2] for x in sorted_record) * prom_quant
            sum_costs = 0
            # DEBUG
            log_msg = (
                f"q = {prom_quant:.2f}, threshold = {cost_threshold:.2f}\n"
                + ", ".join(
                    [
                        f"{x[0]}:{x[2]:.2f}({x[1]:.3f},{int(x[3])})"
                        for x in sorted_record
                    ]
                )
            )
            for id, _, cost, was_promoted in sorted_record:
                sum_costs += cost
                if sum_costs > cost_threshold:
                    log_msg += "\nNothing to promote"
                    break
                if not was_promoted:
                    log_msg += f"\nPromote {id}: sum_costs = {sum_costs:.2f}"
                    ret_id = id
                    break
            logger.debug(log_msg)  # DEBUG
        return ret_id

    def _register_metrics_at_rung_level(self, trial_id, result, recorded):
        metric_value = result[self._metric]
        cost_value = result[self._cost_attr]
        assert trial_id not in recorded  # Sanity check
        recorded[trial_id] = (metric_value, cost_value, False)
