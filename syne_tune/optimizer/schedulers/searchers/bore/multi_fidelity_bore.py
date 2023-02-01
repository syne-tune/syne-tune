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
from typing import Optional, List, Dict, Any
import logging
import numpy as np

from syne_tune.optimizer.schedulers.searchers.bore import Bore

logger = logging.getLogger(__name__)


class MultiFidelityBore(Bore):
    """
    Adapts BORE (Tiao et al.) for the multi-fidelity Hyperband setting following
    BOHB (Falkner et al.). Once we collected enough data points on the smallest
    resource level, we fit a probabilistic classifier and sample from it until we have
    a sufficient amount of data points for the next higher resource level. We then
    refit the classifier on the data of this resource level. These steps are
    iterated until we reach the highest resource level. References:

        | BORE: Bayesian Optimization by Density-Ratio Estimation,
        | Tiao, Louis C and Klein, Aaron and Seeger, Matthias W and Bonilla, Edwin V. and Archambeau, Cedric and Ramos, Fabio
        | Proceedings of the 38th International Conference on Machine Learning

    and

        | BOHB: Robust and Efficient Hyperparameter Optimization at Scale
        | S. Falkner and A. Klein and F. Hutter
        | Proceedings of the 35th International Conference on Machine Learning

    Additional arguments on top of parent class
    :class:`~syne_tune.optimizer.schedulers.searchers.bore.Bore`:

    :param resource_attr: Name of resource attribute. Defaults to "epoch"
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        points_to_evaluate: Optional[List[dict]] = None,
        allow_duplicates: Optional[bool] = None,
        mode: Optional[str] = None,
        gamma: Optional[float] = None,
        calibrate: Optional[bool] = None,
        classifier: Optional[str] = None,
        acq_optimizer: Optional[str] = None,
        feval_acq: Optional[int] = None,
        random_prob: Optional[float] = None,
        init_random: Optional[int] = None,
        classifier_kwargs: Optional[dict] = None,
        resource_attr: str = "epoch",
        **kwargs,
    ):
        if acq_optimizer is None:
            acq_optimizer = "rs_with_replacement"
        super().__init__(
            config_space,
            metric=metric,
            points_to_evaluate=points_to_evaluate,
            allow_duplicates=allow_duplicates,
            mode=mode,
            gamma=gamma,
            calibrate=calibrate,
            classifier=classifier,
            acq_optimizer=acq_optimizer,
            feval_acq=feval_acq,
            random_prob=random_prob,
            init_random=init_random,
            classifier_kwargs=classifier_kwargs,
            **kwargs,
        )
        self.resource_attr = resource_attr
        self.resource_levels = []

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers.multi_fidelity import (
            MultiFidelitySchedulerMixin,
        )

        super().configure_scheduler(scheduler)
        assert isinstance(
            scheduler, MultiFidelitySchedulerMixin
        ), "This searcher requires MultiFidelitySchedulerMixin scheduler"
        self.resource_attr = scheduler.resource_attr

    def _train_model(self, train_data: np.ndarray, train_targets: np.ndarray) -> bool:
        # find the highest resource level we have at least one data points of the positive class
        min_data_points = int(1 / self.gamma)
        unique_resource_levels, counts = np.unique(
            self.resource_levels, return_counts=True
        )
        idx = np.where(counts >= min_data_points)[0]

        if len(idx) == 0:
            return False

        # collect data on the highest resource level
        highest_resource_level = unique_resource_levels[idx[-1]]
        indices = np.where(self.resource_levels == highest_resource_level)[0]

        train_data = np.array([self.inputs[i] for i in indices])
        train_targets = np.array([self.targets[i] for i in indices])

        return super()._train_model(train_data, train_targets)

    def _update(self, trial_id: str, config: Dict[str, Any], result: Dict[str, Any]):
        super()._update(trial_id=trial_id, config=config, result=result)
        resource_level = int(result[self.resource_attr])
        self.resource_levels.append(resource_level)
