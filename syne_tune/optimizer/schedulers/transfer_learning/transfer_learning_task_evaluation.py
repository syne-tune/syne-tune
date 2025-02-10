from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


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
        self, k: int, objective: str, do_minimize: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Returns the best k hyperparameter configurations.
        :param k: The number of top hyperparameters to return.
        :param do_minimize:  indicating if we minimize or maximize the optimization problem.
        :param objective: The objective to consider for ranking hyperparameters.
        :returns: List of hyperparameters in order.
        """
        assert k > 0 and isinstance(k, int), f"{k} is no positive integer."
        assert objective in self.objectives_names, f"Unknown objective {objective}."

        # average over seed and take best fidelity
        avg_objective = self.objective_values(objective_name=objective).mean(axis=1)

        if do_minimize:
            avg_objective = avg_objective.min(axis=1)
        else:
            avg_objective = avg_objective.max(axis=1)
        best_hp_task_indices = avg_objective.argsort()
        if not do_minimize:
            best_hp_task_indices = best_hp_task_indices[::-1]
        return self.hyperparameters.loc[best_hp_task_indices[:k]].to_dict("records")
