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
import logging
from typing import Dict, Callable, Optional

import pandas as pd

from syne_tune.optimizer.scheduler import TrialScheduler
from syne_tune.optimizer.schedulers.transfer_learning import (
    TransferLearningMixin,
    TransferLearningTaskEvaluations,
)
from syne_tune.config_space import (
    Categorical,
    restrict_domain,
    choice,
    config_space_size,
)


class BoundingBox(TransferLearningMixin, TrialScheduler):
    def __init__(
        self,
        scheduler_fun: Callable[[Dict, str, str], TrialScheduler],
        config_space: Dict,
        metric: str,
        transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
        mode: Optional[str] = "min",
        num_hyperparameters_per_task: int = 1,
    ):
        """
        Simple baseline that computes a bounding-box of the best candidate found in previous tasks to restrict the
         search space to only good candidates. The bounding-box is obtained by restricting to the min-max of best
         numerical hyperparameters and restricting to the set of best candidates on categorical parameters.

        Reference: Learning search spaces for Bayesian optimization: Another view of hyperparameter transfer learning.
        Valerio Perrone, Huibin Shen, Matthias Seeger, Cédric Archambeau, Rodolphe Jenatton. Neurips 2019.

        :param scheduler_fun: function that takes a configuration space (Dict), a mode (str) and a metric (str)
        and returns a scheduler. This is required since the final configuration space is known only after computing
        a bounding-box. For instance,
        `scheduler_fun=lambda new_config_space, mode, metric: RandomSearch(new_config_space, metric, mode)`
        will consider a random-search on the config-space is restricted to the bounding of best evaluations of previous
        tasks.
        :param config_space: initial search-space to consider, will be updated to the bounding of best evaluations of
        previous tasks
        :param metric: objective name to optimize, must be present in transfer learning evaluations.
        :param transfer_learning_evaluations: dictionary from task name to offline evaluations.
        :param mode: mode to be considered, default to min.
        :param num_hyperparameters_per_task: number of best hyperparameter to take per task when computing the bounding
        box, default to 1.
        """
        super().__init__(
            config_space=config_space,
            transfer_learning_evaluations=transfer_learning_evaluations,
            metric_names=[metric],
        )
        assert mode in ["min", "max"], "mode must be either 'min' or 'max'."

        config_space = self.compute_box(
            config_space=config_space,
            transfer_learning_evaluations=transfer_learning_evaluations,
            mode=mode,
            num_hyperparameters_per_task=num_hyperparameters_per_task,
            metric=metric,
        )
        print(f"hyperparameter ranges of best previous configurations {config_space}")
        print(f"({config_space_size(config_space)} options)")
        self.scheduler = scheduler_fun(config_space, mode, metric)

    def compute_box(
        self,
        config_space: Dict,
        transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
        mode: str,
        num_hyperparameters_per_task: int,
        metric: str,
    ) -> Dict:
        top_k_per_task = self.top_k_hyperparameter_configurations_per_task(
            transfer_learning_evaluations=transfer_learning_evaluations,
            num_hyperparameters_per_task=num_hyperparameters_per_task,
            mode=mode,
            metric=metric,
        )
        hp_df = pd.DataFrame(
            [hp for _, top_k_hp in top_k_per_task.items() for hp in top_k_hp]
        )

        # compute bounding-box on all hyperparameters that are numerical or categorical
        new_config_space = {}
        for i, (name, domain) in enumerate(config_space.items()):
            if hasattr(domain, "sample"):
                if isinstance(domain, Categorical):
                    hp_values = list(sorted(hp_df.loc[:, name].unique()))
                    new_config_space[name] = choice(hp_values)
                elif hasattr(domain, "lower") and hasattr(domain, "upper"):
                    # domain is numerical, set new lower and upper ranges with bounding-box values
                    new_config_space[name] = restrict_domain(
                        numerical_domain=domain,
                        lower=hp_df.loc[:, name].min(),
                        upper=hp_df.loc[:, name].max(),
                    )
                else:
                    # no known way to compute bounding over non numerical domains such as functional
                    new_config_space[name] = domain
            else:
                new_config_space[name] = domain
        logging.info(
            f"new configuration space obtained after computing bounding-box: {new_config_space}"
        )

        return new_config_space

    def suggest(self, *args, **kwargs):
        return self.scheduler.suggest(*args, **kwargs)

    def on_trial_add(self, *args, **kwargs):
        self.scheduler.on_trial_add(*args, **kwargs)

    def on_trial_complete(self, *args, **kwargs):
        self.scheduler.on_trial_complete(*args, **kwargs)

    def on_trial_remove(self, *args, **kwargs):
        self.scheduler.on_trial_remove(*args, **kwargs)

    def on_trial_error(self, *args, **kwargs):
        self.scheduler.on_trial_error(*args, **kwargs)

    def on_trial_result(self, *args, **kwargs) -> str:
        return self.scheduler.on_trial_result(*args, **kwargs)

    def metric_mode(self) -> str:
        return self.scheduler.metric_mode()
