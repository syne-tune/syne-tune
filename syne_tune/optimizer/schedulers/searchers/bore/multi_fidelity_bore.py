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
from typing import Dict, Optional, List
import logging
import numpy as np

from syne_tune.optimizer.schedulers.searchers.bore import Bore

__all__ = ['MultiFidelityBore']

logger = logging.getLogger(__name__)


class MultiFidelityBore(Bore):
    """

    """

    def __init__(
            self,
            config_space: dict, metric: str,
            points_to_evaluate=None,
            random_seed=None, mode: str = 'max', gamma: float = 0.25,
            calibrate: bool = False, classifier: str = 'xgboost',
            acq_optimizer: str = 'rs_with_replacement', feval_acq: int = 500,
            random_prob: float = 0.0, init_random: int = 6,
            classifier_kwargs: dict = None,
            resource_attr: str = 'epoch',
            **kwargs):
        super().__init__(
            config_space,
            metric=metric,
            points_to_evaluate=points_to_evaluate,
            mode=mode,
            random_seed=random_seed,
            gamma=gamma,
            calibrate=calibrate,
            classifier=classifier,
            acq_optimizer=acq_optimizer,
            feval_acq=feval_acq,
            random_prob = random_prob,
            init_random = init_random,
            classifier_kwargs = classifier_kwargs,
            **kwargs)

        self.resource_attr = resource_attr
        self.resource_levels = []

        # self.num_min_data_points = len(self._hp_ranges) if num_min_data_points is None else num_min_data_points

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers.hyperband import \
            HyperbandScheduler
        from syne_tune.optimizer.schedulers.synchronous.hyperband import \
            SynchronousHyperbandScheduler

        assert isinstance(scheduler, HyperbandScheduler) or \
               isinstance(scheduler, SynchronousHyperbandScheduler), \
            "This searcher requires HyperbandScheduler or " +\
            "SynchronousHyperbandScheduler scheduler"

    def train_model(self, train_data, train_targets):

        # find the highest resource level we have at least one data points of the positive class
        min_data_points = int(1 / self.gamma)
        unique_resource_levels, counts = np.unique(self.resource_levels, return_counts=True)
        idx = np.where(counts >= min_data_points)[0]

        if len(idx) == 0:
            return

        # collect data on the highest resource level
        highest_resource_level = unique_resource_levels[idx[-1]]
        indices = np.where(self.resource_levels == highest_resource_level)[0]

        train_data = np.array([self.inputs[i] for i in indices])
        train_targets = np.array([self.targets[i] for i in indices])

        super().train_model(train_data, train_targets)

    def _update(self, trial_id: str, config: Dict, result: Dict):
        super()._update(trial_id=trial_id, config=config, result=result)
        resource_level = int(result[self.resource_attr])
        self.resource_levels.append(resource_level)
