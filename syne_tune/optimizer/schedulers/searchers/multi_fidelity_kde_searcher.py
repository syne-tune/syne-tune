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

from syne_tune.optimizer.schedulers.searchers.kde_searcher import KernelDensityEstimator

__all__ = ['MultiFidelityKernelDensityEstimator']

logger = logging.getLogger(__name__)


class MultiFidelityKernelDensityEstimator(KernelDensityEstimator):

    def __init__(
            self,
            configspace: Dict,
            metric: str,
            num_init_random_draws: int = 5,
            mode: str = "min",
            num_min_data_points: int = None,
            top_n_percent: int = 15,
            min_bandwidth: float = 0.1,
            num_candidates: int = 64,
            bandwidth_factor: int = 3,
            random_fraction: float = .33,
            resource_attr: str = 'epochs',
            points_to_evaluate: Optional[List[Dict]] = None,
            **kwargs
    ):
        super().__init__(configspace, metric, num_init_random_draws, mode, num_min_data_points,
                         top_n_percent, min_bandwidth, num_candidates, bandwidth_factor, random_fraction,
                         points_to_evaluate, **kwargs)

        self.resource_attr = resource_attr

        self.resource_levels = []

    def _fit_kde_on_highest_resource_level(self, config, result):
        resource_level = result[self.resource_attr]
        self.resource_levels.append(resource_level)

        self.X.append(self.to_feature(
            config=config,
            configspace=self.configspace,
            categorical_maps=self.categorical_maps,
        ))
        self.y.append(self.to_objective(result))

        unique_resource_levels, counts = np.unique(self.resource_levels, return_counts=True)
        idx = np.where(counts >= self.num_min_data_points)[0]
        if len(idx) == 0:
            return

        highest_resource_level = unique_resource_levels[idx[-1]]
        indices = np.where(self.resource_levels == highest_resource_level)[0]

        train_data = np.array([self.X[i] for i in indices])
        train_targets = np.array([self.y[i] for i in indices])

        self.train_kde(train_data, train_targets)

    def _update(self, trial_id: str, config: Dict, result: Dict):
        self._fit_kde_on_highest_resource_level(config, result)
