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
from typing import Dict
import logging
import numpy as np

from syne_tune.optimizer.schedulers.searchers.bore import Bore

__all__ = ["MultiFidelityBore"]

logger = logging.getLogger(__name__)


class MultiFidelityBore(Bore):
    """
    Adapts BORE (Tiao et al.) for the multi-fidelity Hyperband setting following Falkner et al. Once we collected enough
    data points on the smallest resource level, we fit a probabilistic classifier and sample from it until we have
    a sufficient amount of data points for the next higher resource level. We then refit the classifer on the data of
    this resource level. These steps are iterated until we reach the highest resource level.


    BORE: Bayesian Optimization by Density-Ratio Estimation,
    Tiao, Louis C and Klein, Aaron and Seeger, Matthias W and Bonilla, Edwin V. and Archambeau, Cedric and Ramos, Fabio
    Proceedings of the 38th International Conference on Machine Learning

    BOHB: Robust and Efficient Hyperparameter Optimization at Scale
    S. Falkner and A. Klein and F. Hutter
    Proceedings of the 35th International Conference on Machine Learning

    :param config_space: Configuration space. Constant parameters are filtered out
    :param metric: Name of metric reported by evaluation function.
    :param points_to_evaluate:
    :param gamma: Defines the percentile, i.e how many percent of configuration are used to model l(x).
    :param calibrate: If set to true, we calibrate the predictions of the classifier via CV
    :param classifier: The binary classifier to model the acquisition function.
        Choices: {'mlp', 'gp', 'xgboost', 'rf}
    :param random_seed: seed for the random number generator
    :param acq_optimizer: The optimization method to maximize the acquisition function. Choices: {'de', 'rs'}
    :param feval_acq: Maximum allowed function evaluations of the acquisition function.
    :param random_prob: probability for returning a random configurations (epsilon greedy)
    :param init_random: Number of initial random configurations before we start with the optimization.
    :param classifier_kwargs: Dict that contains all hyperparameters for the classifier
    """

    def __init__(
        self,
        config_space: dict,
        metric: str,
        points_to_evaluate=None,
        random_seed=None,
        mode: str = "max",
        gamma: float = 0.25,
        calibrate: bool = False,
        classifier: str = "xgboost",
        acq_optimizer: str = "rs_with_replacement",
        feval_acq: int = 500,
        random_prob: float = 0.0,
        init_random: int = 6,
        classifier_kwargs: dict = None,
        resource_attr: str = "epoch",
        **kwargs
    ):
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
            random_prob=random_prob,
            init_random=init_random,
            classifier_kwargs=classifier_kwargs,
            **kwargs
        )

        self.resource_attr = resource_attr
        self.resource_levels = []

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
        from syne_tune.optimizer.schedulers.synchronous.hyperband import (
            SynchronousHyperbandScheduler,
        )

        super().configure_scheduler(scheduler)
        assert isinstance(scheduler, HyperbandScheduler) or isinstance(
            scheduler, SynchronousHyperbandScheduler
        ), (
            "This searcher requires HyperbandScheduler or "
            + "SynchronousHyperbandScheduler scheduler"
        )

    def train_model(self, train_data, train_targets):
        # find the highest resource level we have at least one data points of the positive class
        min_data_points = int(1 / self.gamma)
        unique_resource_levels, counts = np.unique(
            self.resource_levels, return_counts=True
        )
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
