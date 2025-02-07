from typing import Any, Dict, List

from syne_tune.optimizer.schedulers.transfer_learning.transfer_learning_task_evaluation import (
    TransferLearningTaskEvaluations,
)


class TransferLearningMixin:
    def __init__(
        self,
        config_space: Dict,
        transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
        metric: str,
        random_seed: int = None,
        **kwargs,
    ):
        """
        A mixin that adds basic functionality for using offline evaluations.
        :param config_space: configuration space to be sampled from
        :param transfer_learning_evaluations: dictionary from task name to offline evaluations.
        :param metric_names: name of the metric to be optimized.
        """
        super().__init__(
            config_space=config_space, random_seed=random_seed, metric=metric
        )
        self.metric = metric
        self._check_consistency(
            config_space=config_space,
            transfer_learning_evaluations=transfer_learning_evaluations,
        )

    def _check_consistency(
        self,
        config_space: Dict,
        transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
    ):
        for task, evals in transfer_learning_evaluations.items():
            for key in config_space.keys():
                assert key in evals.hyperparameters.columns, (
                    f"the key {key} of the config space should appear in transfer learning evaluations "
                    f"hyperparameters {evals.hyperparameters.columns}"
                )
            assert self.metric in evals.objectives_names, (
                f"all objectives used in the scheduler {self.metric} should appear in transfer learning "
                f"evaluations objectives {evals.objectives_names}"
            )

    def top_k_hyperparameter_configurations_per_task(
        self,
        transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
        num_hyperparameters_per_task: int,
        do_minimize: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns the best hyperparameter configurations for each task.
        :param transfer_learning_evaluations: Set of candidates to choose from.
        :param num_hyperparameters_per_task: The number of top hyperparameters per task to return.
        :param do_minimize: indicating if the optimization problem is minimized.
        :returns: Dict which maps from task name to list of hyperparameters in order.
        """
        assert num_hyperparameters_per_task > 0 and isinstance(
            num_hyperparameters_per_task, int
        ), f"{num_hyperparameters_per_task} is no positive integer."

        best_hps = dict()
        for task, evaluation in transfer_learning_evaluations.items():
            best_hps[task] = evaluation.top_k_hyperparameter_configurations(
                num_hyperparameters_per_task, self.metric, do_minimize
            )
        return best_hps
