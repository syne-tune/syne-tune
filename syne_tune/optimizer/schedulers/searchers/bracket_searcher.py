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
import numpy as np

from syne_tune.optimizer.schedulers.searchers.searcher import (
    BaseSearcher,
    RandomSearcher,
    GridSearcher,
)


class SearcherWithDistributionOverBrackets(BaseSearcher):
    """
    Base class for searchers which provide a method to return a distribution
    over brackets. This is used in multi-fidelity schedulers like
    :class:`HyperbandScheduler` in order to sample the bracket (or
    effective `grace_period`) for a new trial.

    Also provides a callback which is called in `_update`, which allows the
    child class to update the distribution over brackets, given new data.
    """

    def __init__(self, config_space, metric, points_to_evaluate=None):
        super().__init__(
            config_space, metric=metric, points_to_evaluate=points_to_evaluate
        )

    def distribution_over_brackets(self) -> np.ndarray:
        raise NotImplementedError


class DefaultHyperbandBracketSamplingSearcher(SearcherWithDistributionOverBrackets):
    """
    Implements default bracket distribution, where probability for each bracket
    is proportional to the number of slots in each bracket in synchronous
    Hyperband.
    """

    def __init__(self, config_space, metric, points_to_evaluate=None, **kwargs):
        super().__init__(
            config_space, metric=metric, points_to_evaluate=points_to_evaluate
        )
        # Delayed initialization in `configure_scheduler`
        self.num_brackets = None
        self.rung_levels = None
        self._distribution = None

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers import HyperbandScheduler

        super().configure_scheduler(scheduler)
        assert isinstance(
            scheduler, HyperbandScheduler
        ), "Scheduler must be HyperbandScheduler"
        self.num_brackets = scheduler.terminator.num_brackets
        self.rung_levels = scheduler.rung_levels
        self._set_distribution()

    def distribution_over_brackets(self) -> np.ndarray:
        assert self._distribution is not None, "Call 'configure_scheduler' first"
        return self._distribution

    def _set_distribution(self):
        if self.num_brackets > 1:
            smax_plus1 = len(self.rung_levels)
            assert self.num_brackets <= smax_plus1
            self._distribution = np.array(
                [
                    smax_plus1 / ((smax_plus1 - s) * self.rung_levels[s])
                    for s in range(self.num_brackets)
                ]
            )
            self._distribution /= self._distribution.sum()
        else:
            self._distribution = np.ones(1)


class RandomWithDefaultBracketSamplingSearcher(
    DefaultHyperbandBracketSamplingSearcher, RandomSearcher
):
    def __init__(self, config_space, metric, points_to_evaluate=None, **kwargs):
        super().__init__(config_space, metric, points_to_evaluate, **kwargs)

    def configure_scheduler(self, scheduler):
        DefaultHyperbandBracketSamplingSearcher.configure_scheduler(self, scheduler)
        RandomSearcher.configure_scheduler(self, scheduler)


class GridWithDefaultBracketSamplingSearcher(
    DefaultHyperbandBracketSamplingSearcher, GridSearcher
):
    def __init__(self, config_space, metric, points_to_evaluate=None, **kwargs):
        super().__init__(config_space, metric, points_to_evaluate, **kwargs)

    def configure_scheduler(self, scheduler):
        DefaultHyperbandBracketSamplingSearcher.configure_scheduler(self, scheduler)
        GridSearcher.configure_scheduler(self, scheduler)
