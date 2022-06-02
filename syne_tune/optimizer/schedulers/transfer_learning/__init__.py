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
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

__all__ = [
    "TransferLearningTaskEvaluations",
    "TransferLearningMixin",
    "BoundingBox",
    "RUSHScheduler",
]


@dataclass
class TransferLearningTaskEvaluations:
    """Class that contains offline evaluations for a task that can be used for transfer learning.
    Args:
        configuration_space: Dict the configuration space that was used when sampling evaluations.
        hyperparameters: pd.DataFrame the hyperparameters values that were acquired, all keys of configuration-space
         should appear as columns.
        objectives_names: List[str] the name of the objectives that were acquired
        objectives_evaluations: np.array values of recorded objectives, must have shape
            (num_evals, num_seeds, num_fidelities, num_objectives)
    """

    configuration_space: Dict
    hyperparameters: pd.DataFrame
    objectives_names: List[str]
    objectives_evaluations: np.array

    def __post_init__(self):
        assert len(self.objectives_names) == self.objectives_evaluations.shape[-1]
        assert len(self.hyperparameters) == self.objectives_evaluations.shape[0]
        assert self.objectives_evaluations.ndim == 4, (
            "objective evaluations should be of shape "
            "(num_evals, num_seeds, num_fidelities, num_objectives)"
        )
        for col in self.hyperparameters.keys():
            assert col in self.configuration_space

    def objective_values(self, objective_name: str) -> np.array:
        return self.objectives_evaluations[
            ..., self.objective_index(objective_name=objective_name)
        ]

    def objective_index(self, objective_name: str) -> int:
        matches = [
            i for i, name in enumerate(self.objectives_names) if name == objective_name
        ]
        assert len(matches) >= 1, (
            f"could not find objective {objective_name} in recorded objectives "
            f"{self.objectives_names}"
        )
        return matches[0]

    def top_k_hyperparameter_configurations(
        self, k: int, mode: str, objective: str
    ) -> List[Dict[str, Any]]:
        """
        Returns the best k hyperparameter configurations.
        :param k: The number of top hyperparameters to return.
        :param mode: 'min' or 'max', indicating the type of optimization problem.
        :param objective: The objective to consider for ranking hyperparameters.
        :returns: List of hyperparameters in order.
        """
        assert k > 0 and isinstance(k, int), f"{k} is no positive integer."
        assert mode in ["min", "max"], f"Unknown mode {mode}, must be 'min' or 'max'."
        assert objective in self.objectives_names, f"Unknown objective {objective}."

        # average over seed and take best fidelity
        avg_objective = self.objective_values(objective_name=objective).mean(axis=1)
        if mode == "max":
            avg_objective = avg_objective.max(axis=1)
        else:
            avg_objective = avg_objective.min(axis=1)
        best_hp_task_indices = avg_objective.argsort()
        if mode == "max":
            best_hp_task_indices = best_hp_task_indices[::-1]
        return self.hyperparameters.loc[best_hp_task_indices[:k]].to_dict("records")


class TransferLearningMixin:
    def __init__(
        self,
        config_space: Dict,
        transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
        metric_names: List[str],
        **kwargs,
    ):
        """
        A mixin that adds basic functionality for using offline evaluations.
        :param config_space: configuration space to be sampled from
        :param transfer_learning_evaluations: dictionary from task name to offline evaluations.
        :param metric_names: name of the metric to be optimized.
        """
        super().__init__(config_space=config_space, **kwargs)
        self._metric_names = metric_names
        self._check_consistency(
            config_space=config_space,
            transfer_learning_evaluations=transfer_learning_evaluations,
            metric_names=metric_names,
        )

    def _check_consistency(
        self,
        config_space: Dict,
        transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
        metric_names: List[str],
    ):
        for task, evals in transfer_learning_evaluations.items():
            for key in config_space.keys():
                assert key in evals.hyperparameters.columns, (
                    f"the key {key} of the config space should appear in transfer learning evaluations "
                    f"hyperparameters {evals.hyperparameters.columns}"
                )
            assert all([m in evals.objectives_names for m in metric_names]), (
                f"all objectives used in the scheduler {self.metric_names()} should appear in transfer learning "
                f"evaluations objectives {evals.objectives_names}"
            )

    def metric_names(self) -> List[str]:
        return self._metric_names

    def top_k_hyperparameter_configurations_per_task(
        self,
        transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
        num_hyperparameters_per_task: int,
        mode: str,
        metric: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns the best hyperparameter configurations for each task.
        :param transfer_learning_evaluations: Set of candidates to choose from.
        :param num_hyperparameters_per_task: The number of top hyperparameters per task to return.
        :param mode: 'min' or 'max', indicating the type of optimization problem.
        :param metric: The metric to consider for ranking hyperparameters.
        :returns: Dict which maps from task name to list of hyperparameters in order.
        """
        assert num_hyperparameters_per_task > 0 and isinstance(
            num_hyperparameters_per_task, int
        ), f"{num_hyperparameters_per_task} is no positive integer."
        assert mode in ["min", "max"], f"Unknown mode {mode}, must be 'min' or 'max'."
        assert metric in self.metric_names(), f"Unknown metric {metric}."
        best_hps = dict()
        for task, evaluation in transfer_learning_evaluations.items():
            best_hps[task] = evaluation.top_k_hyperparameter_configurations(
                num_hyperparameters_per_task, mode, metric
            )
        return best_hps


from syne_tune.optimizer.schedulers.transfer_learning.bounding_box import BoundingBox
from syne_tune.optimizer.schedulers.transfer_learning.rush import RUSHScheduler
