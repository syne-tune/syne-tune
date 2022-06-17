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
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    INTERNAL_METRIC_NAME,
)


class SkipOptimizationPredicate:
    """
    Interface for skip_optimization predicate in
    :class:`ModelStateTransformer`.

    """

    def reset(self):
        """
        If there is an internal state, reset it to its initial value
        """
        pass

    def __call__(self, state: TuningJobState) -> bool:
        """
        :param state: Current TuningJobState
        :return: Skip hyperparameter optimization?
        """
        raise NotImplementedError


class NeverSkipPredicate(SkipOptimizationPredicate):
    """
    Hyperparameter optimization is never skipped.

    """

    def __call__(self, state: TuningJobState) -> bool:
        return False


class AlwaysSkipPredicate(SkipOptimizationPredicate):
    """
    Hyperparameter optimization is always skipped.

    """

    def __call__(self, state: TuningJobState) -> bool:
        return True


class SkipPeriodicallyPredicate(SkipOptimizationPredicate):
    """
    Let N be the number of labeled points for metric `metric_name`.
    Optimizations are not skipped if N < init_length. Afterwards,
    we increase a counter whenever N is larger than in the previous
    call. With respect to this counter, optimizations are done every
    period times, in between they are skipped.

    """

    def __init__(
        self, init_length: int, period: int, metric_name: str = INTERNAL_METRIC_NAME
    ):
        assert init_length >= 0
        assert period > 1
        self.init_length = init_length
        self.period = period
        self.metric_name = metric_name
        self.reset()

    def reset(self):
        self._counter = 0
        # Need to make sure that if called several times with the same state,
        # we return the same value
        self._last_size = None
        self._last_retval = None

    def __call__(self, state: TuningJobState) -> bool:
        num_labeled = state.num_observed_cases(self.metric_name)
        if num_labeled == self._last_size:
            return self._last_retval
        if self._last_size is not None:
            assert (
                num_labeled > self._last_size
            ), "num_labeled = {} < {} = _last_size".format(num_labeled, self._last_size)
        if num_labeled < self.init_length:
            ret_value = False
        else:
            ret_value = self._counter % self.period != 0
            self._counter += 1
        self._last_size = num_labeled
        self._last_size = ret_value
        return ret_value


class SkipNoMaxResourcePredicate(SkipOptimizationPredicate):
    """
    This predicate works for multi-fidelity HPO, see for example
    :class:`GPMultiFidelitySearcher`.

    We track the number of labeled datapoints at resource level max_resource.
    HP optimization is skipped if the total number of labeled cases is >=
    init_length, and if the number of max_resource cases has not increased
    since the last recent optimization.

    This means that as long as the dataset only grows w.r.t. cases at lower
    resources than max_resource, this does not trigger HP optimization.

    """

    def __init__(
        self,
        init_length: int,
        max_resource: int,
        metric_name: str = INTERNAL_METRIC_NAME,
    ):
        assert init_length >= 0
        self.init_length = init_length
        self.metric_name = metric_name
        self.max_resource = str(max_resource)
        self.reset()

    def reset(self):
        self.lastrec_max_resource_cases = None

    def _num_max_resource_cases(self, state: TuningJobState):
        def is_max_resource(metrics: dict) -> int:
            return int(self.max_resource in metrics[self.metric_name])

        return sum(is_max_resource(ev.metrics) for ev in state.trials_evaluations)

    def __call__(self, state: TuningJobState) -> bool:
        if state.num_observed_cases(self.metric_name) < self.init_length:
            return False
        num_max_resource_cases = self._num_max_resource_cases(state)
        if (
            self.lastrec_max_resource_cases is None
            or num_max_resource_cases > self.lastrec_max_resource_cases
        ):
            self.lastrec_max_resource_cases = num_max_resource_cases
            return False
        else:
            return True
