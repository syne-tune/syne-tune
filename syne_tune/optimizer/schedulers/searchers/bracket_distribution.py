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

from syne_tune.optimizer.scheduler import TrialScheduler


class BracketDistribution:
    """
    Configures multi-fidelity schedulers such as :class:`HyperbandScheduler` with
    distribution over brackets. This distribution can be fixed up front, or
    change adaptively during the course of an experiment.

    TODO: Support for adaptive update (needed for Hyper-Tune)
    """

    def __call__(self) -> np.ndarray:
        """
        :return: Distribution over brackets
        """
        raise NotImplementedError

    def configure(self, scheduler: TrialScheduler):
        """
        This method is called in by the scheduler just after
        `self.searcher.configure_scheduler`. The searcher must be accessible
        via `self.searcher`.
        The `__call__` method cannot be used before this method has been
        called.
        """
        raise NotImplementedError


class DefaultHyperbandBracketDistribution(BracketDistribution):
    """
    Implements default bracket distribution, where probability for each bracket
    is proportional to the number of slots in each bracket in synchronous
    Hyperband.
    """

    def __init__(self):
        self.num_brackets = None
        self.rung_levels = None
        self._distribution = None

    def configure(self, scheduler: TrialScheduler):
        from syne_tune.optimizer.schedulers import HyperbandScheduler

        assert isinstance(
            scheduler, HyperbandScheduler
        ), "Scheduler must be HyperbandScheduler"
        self.num_brackets = scheduler.terminator.num_brackets
        self.rung_levels = scheduler.rung_levels
        self._set_distribution()

    def __call__(self) -> np.ndarray:
        assert self._distribution is not None, "Call 'configure' first"
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
