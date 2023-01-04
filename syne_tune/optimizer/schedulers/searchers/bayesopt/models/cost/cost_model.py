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
from typing import List
from dataclasses import dataclass

from syne_tune.optimizer.schedulers.searchers.utils.common import Configuration
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)

__all__ = ["CostValue", "CostModel"]


@dataclass
class CostValue:
    """
    Represents cost value :math:`(c_0(x), c_1(x))`:

    * :math:`c_0(x)`: Startup cost for evaluation at config :math:`x`
    * :math:`c_1(x)`: Cost per unit of resource :math:`r` at config :math:`x`

    Our assumption is that, under the model, an evaluation at :math:`x` until
    resource level :math:`r = 1, 2, 3, \dots` costs
    :math:`c(x, r) = c_0(x) + r c_1(x)`
    """

    c0: float
    c1: float


class CostModel:
    """
    Interface for (temporal) cost model in the context of multi-fidelity HPO.
    We assume there are configurations :math:`x` and resource levels :math:`r`
    (for example, number of epochs). Here, :math:`r` is a positive int.
    Can be seen as simplified version of surrogate model, which is mainly used
    in order to draw (jointly dependent) values from the posterior over
    cost values :math:`(c_0(x), c_1(x))`.

    Note: The model may be random (in which case joint samples are drawn from
    the posterior) or deterministic (in which case the model is fitted to data,
    and then cost values returned are deterministic.

    A cost model has an inner state, which is set by calling :meth:`update`
    passing a dataset. This inner state is then used when :meth:`sample_joint`
    is called.
    """

    @property
    def cost_metric_name(self) -> str:
        """
        :return: Name of metric in :class:`TrialEvaluations` of cases in
            :class:`TuningJobState`
        """
        raise NotImplementedError

    def update(self, state: TuningJobState):
        """
        Update inner representation in order to be ready to return cost value
        samples.

        Note: The metric :attr``cost_metric_name`` must be dict-valued in ``state``,
        with keys being resource values :math:`r`. In order to support a proper
        estimation of :math:`c_0` and :math:`c_1`, there should (ideally) be
        entries with the same :math:`x` and different resource levels :math:`r`.
        The likelihood function takes into account that
        :math:`c(x, r) = c_0(x) + r c_1(x)`.

        :param state: Current dataset (only ``trials_evaluations`` is used)
        """
        raise NotImplementedError

    def resample(self):
        """
        For a random cost model, the state is resampled, such that calls of
        joint_sample before and after are conditionally independent. Normally,
        successive calls of sample_joint are jointly dependent.
        For example, for a linear model, the state resampled here would be the
        weight vector, which is then used in :meth:`sample_joint`.

        For a deterministic cost model, this method does nothing.
        """
        pass

    def sample_joint(self, candidates: List[Configuration]) -> List[CostValue]:
        """
        Draws cost values :math:`(c_0(x), c_1(x))` for candidates (non-extended).

        If the model is random, the sampling is done jointly. Also, if
        :meth:`sample_joint` is called multiple times, the posterior is to be
        updated after each call, such that the sample over the union of
        candidates over all calls is drawn jointly (but see :meth:`resample`).
        Also, if measurement noise is allowed in update, this noise is *not*
        added here. A sample from :math:`c(x, r)` is obtained as
        :math:`c_0(x) + r c_1(x)`. If the model is deterministic, the model
        determined in :meth:`update` is just evaluated.

        :param candidates: Non-extended configs
        :return: List of :math:`(c_0(x), c_1(x))`
        """
        raise NotImplementedError

    @staticmethod
    def event_time(
        start_time: float, level: int, next_milestone: int, cost: CostValue
    ) -> float:
        """
        If a task reported its last recent value at ``start_time`` at level ``level``,
        return time of reaching level ``next_milestone``, given cost ``cost``.

        :param start_time: See above
        :param level: See above
        :param next_milestone: See above
        :param cost: See above
        :return: Time of reaching ``next_milestone`` under cost model
        """
        result = start_time + cost.c1 * (next_milestone - level)
        if level == 0:
            # Add startup time
            result += cost.c0
        return result

    def predict_times(
        self,
        candidates: List[Configuration],
        resources: List[int],
        cost_values: List[CostValue],
        start_time: float = 0,
    ) -> List[float]:
        """
        Given configs :math:`x`, resource values :math:`r` and cost values returned
        by :meth:`sample_joint`, compute time predictions for when each config
        :math:`x` reaches its resource level :math:`r` if started at ``start_time``.

        :param candidates: Configs
        :param resources: Resource levels
        :param cost_values: Cost values from :meth:`sample_joint`
        :param start_time: See above
        :return: Predicted times
        """
        num_cases = len(candidates)
        assert len(resources) == num_cases
        assert len(cost_values) == num_cases
        time_predictions = []
        for candidate, resource, cost in zip(candidates, resources, cost_values):
            time_predictions.append(
                self.event_time(
                    start_time=start_time, level=0, next_milestone=resource, cost=cost
                )
            )
        return time_predictions

    def _check_dataset_has_cost_metric(self, state: TuningJobState):
        assert all(
            self.cost_metric_name in x.metrics for x in state.trials_evaluations
        ), "All labeled cases in state must have metrics[{}]".format(
            self.cost_metric_name
        )
