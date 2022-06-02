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
from typing import Dict, List, Optional

from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.transfer_learning import (
    TransferLearningTaskEvaluations,
    TransferLearningMixin,
)


class RUSHScheduler(TransferLearningMixin, HyperbandScheduler):
    def __init__(
        self,
        config_space: Dict,
        transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
        metric: str,
        type: str = "stopping",
        points_to_evaluate: Optional[List[Dict]] = None,
        custom_rush_points: Optional[List[Dict]] = None,
        num_hyperparameters_per_task: int = 1,
        **kwargs,
    ) -> None:
        """
        A transfer learning variation of Hyperband which uses previously well-performing hyperparameter configurations
        as an initialization. The best hyperparameter configuration of each individual task provided is evaluated.
        The one among them which performs best on the current task will serve as a hurdle and is used to prune
        other candidates. This changes the standard successive halving promotion as follows. As usual, only the top-
        performing fraction is promoted to the next rung level. However, these candidates need to be at least as good
        as the hurdle configuration to be promoted. In practice this means that much fewer candidates can be promoted.

        Reference: A resource-efficient method for repeated HPO and NAS.
        Giovanni Zappella, David Salinas, Cédric Archambeau. AutoML workshop @ ICML 2021.

        :param config_space: configuration space for trial evaluation function.
        :param transfer_learning_evaluations: dictionary from task name to offline evaluations.
        :param metric: objective name to optimize, must be present in transfer learning evaluations.
        :param type: scheduler type ('stopping' or 'promotion'). See :class:`HyperbandScheduler`.
        :param points_to_evaluate: when points_to_evaluate is not None, these configurations are evaluated after
        custom_rush_points and hyperparameter configurations inferred from transfer_learning_evaluations. These points
        are not used to prune any configurations.
        :param custom_rush_points: when custom_rush_points is not None, the provided configurations are evaluated first
        in addition to top performing configurations from other tasks and also serve to preemptively prune
        underperforming configurations
        :param num_hyperparameters_per_task: the number of top hyperparameter configurations to consider per task.
        """
        self._metric_names = [metric]
        assert type in ["stopping", "promotion"], f"Unknown scheduler type {type}"
        top_k_per_task = self.top_k_hyperparameter_configurations_per_task(
            transfer_learning_evaluations=transfer_learning_evaluations,
            num_hyperparameters_per_task=num_hyperparameters_per_task,
            metric=metric,
            mode=kwargs.get("mode", "min"),
        )
        threshold_candidates = [
            hp for _, top_k_hp in top_k_per_task.items() for hp in top_k_hp
        ]
        if custom_rush_points is not None:
            threshold_candidates += custom_rush_points
            threshold_candidates = [
                dict(s) for s in set(frozenset(p.items()) for p in threshold_candidates)
            ]
        num_threshold_candidates = len(threshold_candidates)
        if points_to_evaluate is not None:
            points_to_evaluate = threshold_candidates + [
                hp for hp in points_to_evaluate if hp not in threshold_candidates
            ]
        else:
            points_to_evaluate = threshold_candidates
        super().__init__(
            config_space=config_space,
            transfer_learning_evaluations=transfer_learning_evaluations,
            metric=metric,
            type=f"rush_{type}",
            points_to_evaluate=points_to_evaluate,
            metric_names=[metric],
            rung_system_kwargs={"num_threshold_candidates": num_threshold_candidates},
            **kwargs,
        )
