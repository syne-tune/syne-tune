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
from typing import Dict, Callable, Optional, Any

import pandas as pd

from syne_tune.optimizer.scheduler import TrialScheduler, TrialSuggestion
from syne_tune.backend.trial_status import Trial
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
    """
    Simple baseline that computes a bounding-box of the best candidate found in
    previous tasks to restrict the search space to only good candidates. The
    bounding-box is obtained by restricting to the min-max of the best numerical
    hyperparameters and restricting to the set of the best candidates on categorical
    parameters. Reference:

        | Learning search spaces for Bayesian optimization: Another view of hyperparameter transfer learning.
        | Valerio Perrone, Huibin Shen, Matthias Seeger, Cédric Archambeau, Rodolphe Jenatton.
        | NeurIPS 2019.

    ``scheduler_fun`` is used to create the scheduler to be used here, feeding
    it with the modified config space. Any additional scheduler arguments
    (such as ``points_to_evaluate``) should be encoded inside this function.
    Example:

    .. code-block::

       from syne_tune.optimizer.baselines import RandomSearch

       def scheduler_fun(new_config_space: Dict[str, Any], mode: str, metric: str):
           return RandomSearch(new_config_space, metric, mode)

       bb_scheduler = BoundingBox(scheduler_fun, ...)

    Here, ``bb_scheduler`` represents random search, where the hyperparameter
    ranges are restricted to contain the best evalutions of previous tasks,
    as provided by ``transfer_learning_evaluations``.

    :param scheduler_fun: Maps tuple of configuration space (dict), mode (str),
        metric (str) to a scheduler. This is required since the final
        configuration space is known only after computing a bounding-box.
    :param config_space: Initial configuration space to consider, will be updated
        to the bounding of the best evaluations of previous tasks
    :param metric: Objective name to optimize, must be present in transfer
        learning evaluations.
    :param mode: Mode to be considered, default to "min".
    :param transfer_learning_evaluations: Dictionary from task name to
        offline evaluations.
    :param num_hyperparameters_per_task: Number of the best configurations to
        use per task when computing the bounding box, defaults to 1.
    """

    def __init__(
        self,
        scheduler_fun: Callable[[dict, str, str], TrialScheduler],
        config_space: Dict[str, Any],
        metric: str,
        transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
        mode: Optional[str] = None,
        num_hyperparameters_per_task: int = 1,
    ):
        super().__init__(
            config_space=config_space,
            transfer_learning_evaluations=transfer_learning_evaluations,
            metric_names=[metric],
        )
        if mode is None:
            mode = "min"
        else:
            assert mode in ["min", "max"], "mode must be either 'min' or 'max'."

        config_space = self._compute_box(
            config_space=config_space,
            transfer_learning_evaluations=transfer_learning_evaluations,
            mode=mode,
            num_hyperparameters_per_task=num_hyperparameters_per_task,
            metric=metric,
        )
        print(f"hyperparameter ranges of best previous configurations {config_space}")
        print(f"({config_space_size(config_space)} options)")
        self.scheduler = scheduler_fun(config_space, mode, metric)

    def _compute_box(
        self,
        config_space: Dict[str, Any],
        transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
        mode: str,
        num_hyperparameters_per_task: int,
        metric: str,
    ) -> Dict[str, Any]:
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

    def suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        return self.scheduler.suggest(trial_id)

    def on_trial_add(self, trial: Trial):
        self.scheduler.on_trial_add(trial)

    def on_trial_complete(self, trial: Trial, result: Dict[str, Any]):
        self.scheduler.on_trial_complete(trial, result)

    def on_trial_remove(self, trial: Trial):
        self.scheduler.on_trial_remove(trial)

    def on_trial_error(self, trial: Trial):
        self.scheduler.on_trial_error(trial)

    def on_trial_result(self, trial: Trial, result: Dict[str, Any]) -> str:
        return self.scheduler.on_trial_result(trial, result)

    def metric_mode(self) -> str:
        return self.scheduler.metric_mode()
