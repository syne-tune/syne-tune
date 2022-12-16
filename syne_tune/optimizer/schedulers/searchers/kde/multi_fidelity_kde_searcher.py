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

from syne_tune.optimizer.schedulers.searchers.kde.kde_searcher import (
    KernelDensityEstimator,
)

logger = logging.getLogger(__name__)


class MultiFidelityKernelDensityEstimator(KernelDensityEstimator):
    """
    Adapts :class:`KernelDensityEstimator` to the multi-fidelity setting as proposed
    by Falkner et al such that we can use it with Hyperband. Following Falkner
    et al, we fit the KDE only on the highest resource level where we have at
    least num_min_data_points. Code is based on the implementation by Falkner
    et al: https://github.com/automl/HpBandSter/tree/master/hpbandster

        | BOHB: Robust and Efficient Hyperparameter Optimization at Scale
        | S. Falkner and A. Klein and F. Hutter
        | Proceedings of the 35th International Conference on Machine Learning

    Additional arguments on top of parent class
    :class:`~syne_tune.optimizer.schedulers.searchers.kde.KernelDensityEstimator`:

    :param resource_attr: Name of resource attribute. Defaults to
        ``scheduler.resource_attr`` in :meth:`configure_scheduler`
    """

    def __init__(
        self,
        config_space: dict,
        metric: str,
        points_to_evaluate: Optional[List[dict]] = None,
        mode: Optional[str] = None,
        num_min_data_points: Optional[int] = None,
        top_n_percent: Optional[int] = None,
        min_bandwidth: Optional[float] = None,
        num_candidates: Optional[int] = None,
        bandwidth_factor: Optional[int] = None,
        random_fraction: Optional[float] = None,
        resource_attr: Optional[str] = None,
        **kwargs
    ):
        if min_bandwidth is None:
            min_bandwidth = 0.1
        super().__init__(
            config_space,
            metric=metric,
            points_to_evaluate=points_to_evaluate,
            mode=mode,
            num_min_data_points=num_min_data_points,
            top_n_percent=top_n_percent,
            min_bandwidth=min_bandwidth,
            num_candidates=num_candidates,
            bandwidth_factor=bandwidth_factor,
            random_fraction=random_fraction,
            **kwargs
        )
        self.resource_attr = resource_attr
        self.resource_levels = []

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
        from syne_tune.optimizer.schedulers.synchronous.hyperband import (
            SynchronousHyperbandScheduler,
        )

        assert isinstance(
            scheduler, (HyperbandScheduler, SynchronousHyperbandScheduler)
        ), (
            "This searcher requires HyperbandScheduler or "
            + "SynchronousHyperbandScheduler scheduler"
        )
        self.resource_attr = scheduler.resource_attr

    def _train_kde(self, train_data, train_targets):
        # find the highest resource level we have at least num_min_data_points data points
        unique_resource_levels, counts = np.unique(
            self.resource_levels, return_counts=True
        )
        idx = np.where(counts >= self.num_min_data_points)[0]
        if len(idx) == 0:
            return

        # collect data on the highest resource level
        highest_resource_level = unique_resource_levels[idx[-1]]
        indices = np.where(self.resource_levels == highest_resource_level)[0]

        train_data = np.array([self.X[i] for i in indices])
        train_targets = np.array([self.y[i] for i in indices])

        super()._train_kde(train_data, train_targets)

    def _update(self, trial_id: str, config: Dict, result: Dict):
        super()._update(trial_id=trial_id, config=config, result=result)
        resource_level = int(result[self.resource_attr])
        self.resource_levels.append(resource_level)
