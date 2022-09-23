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
    Represents cost value (c0(x), c1(x)):
        c_0(x): Startup cost for evaluation at config x
        c_1(x): Cost per unit of resource r at config x
    Our assumption is that, under the model, an evaluation at x until resource
    level r = 1, 2, 3, ... costs
        c(x, r) = c_0(x) + r c_1(x)
    """

    c0: float
    c1: float


class CostModel:
    """
    Interface for (temporal) cost model in the context of multi-fidelity HPO.
    We assume there are configurations x and resource levels r (for example,
    r may be number of epochs). Here, r is a positive int.
    Can be seen as simplified version of surrogate model, which is mainly used
    in order to draw (jointly dependent) values from the posterior over
    cost values (c0(x), c1(x)).

    Note: The model may be random (in which case joint samples are drawn from
    the posterior) or deterministic (in which case the model is fitted to data,
    and then cost values returned are deterministic.

    A cost model has an inner state, which is set by calling `update` passing a
    dataset. This inner state is then used when `sample_joint` is called.

    """

    @property
    def cost_metric_name(self) -> str:
        """
        :return: Name of metric in TrialEvaluations of cases in
            TuningJobState
        """
        raise NotImplementedError

    def update(self, state: TuningJobState):
        """
        Update inner representation in order to be ready to return cost value
        samples.

        Note: The metric self.cost_metric_name must be dict-valued in `state`,
        wiht keys being resource values r. In order to support a proper
        estimation of c_0 and c_1, there should (ideally) be entries with the
        same x and different resource levels r. The likelihood function takes
        into account that
            c(x, r) = c_0(x) + r c_1(x)

        :param state: Current dataset (only trials_evaluations is used)
        """
        raise NotImplementedError

    def resample(self):
        """
        For a random cost model, the state is resampled, such that calls of
        joint_sample before and after are conditionally independent. Normally,
        successive calls of sample_joint are jointly dependent.
        For example, for a linear model, the state resampled here would be the
        weight vector, which is then used in 'sample_joint'.

        For a deterministic cost model, this method does nothing.
        """
        pass

    def sample_joint(self, candidates: List[Configuration]) -> List[CostValue]:
        """
        Draws cost values (c_0(x), c_1(x)) for candidates (non-extended).

        If the model is random, the sampling is done jointly. Also, if
        sample_joint is called multiple times, the posterior is to be updated
        after each call, such that the sample over the union of candidates over
        all calls is drawn jointly (but see 'resample'). Also, if measurement
        noise is allowed in update, this noise is *not* added here. A sample
        from c(x, r) is obtained as c_0(x) + r c_1(x).
        If the model is deterministic, the model determined in update is just
        evaluated.

        :param candidates: Non-extended configs
        :return: List of (c_0(x), c_1(x))
        """
        raise NotImplementedError

    @staticmethod
    def event_time(
        start_time: float, level: int, next_milestone: int, cost: CostValue
    ) -> float:
        """
        If a task reported its last recent value at start_time at level level,
        return time of reaching level next_milestone, given cost cost.

        :param start_time:
        :param level:
        :param next_milestone:
        :param cost:
        :return: Time of reaching next_milestone under cost model
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
        Given configs x, resource values r and cost values from sample_joint,
        compute time predictions for when each config x reaches its resource
        level r if started at start_time.

        :param candidates: Configs
        :param resources: Resource levels
        :param cost_values: Cost values from sample_joint
        :param start_time:
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
