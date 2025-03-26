from typing import Dict, List, Optional, Any

from syne_tune.optimizer.schedulers.legacy_hyperband import LegacyHyperbandScheduler
from syne_tune.optimizer.schedulers.transfer_learning import (
    LegacyTransferLearningTaskEvaluations,
    LegacyTransferLearningMixin,
)


class RUSHScheduler(LegacyTransferLearningMixin, LegacyHyperbandScheduler):
    """
    A transfer learning variation of Hyperband which uses previously
    well-performing hyperparameter configurations as an initialization. The best
    hyperparameter configuration of each individual task provided is evaluated.
    The one among them which performs best on the current task will serve as a
    hurdle and is used to prune other candidates. This changes the standard
    successive halving promotion as follows. As usual, only the top-performing
    fraction is promoted to the next rung level. However, these candidates need
    to be at least as good as the hurdle configuration to be promoted. In practice
    this means that much fewer candidates can be promoted. Reference:

        | A resource-efficient method for repeated HPO and NAS.
        | Giovanni Zappella, David Salinas, Cédric Archambeau.
        | AutoML workshop @ ICML 2021.

    Additional arguments on top of parent class
    :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.

    :param transfer_learning_evaluations: Dictionary from task name to offline
        evaluations.
    :param points_to_evaluate: If given, these configurations are evaluated
        after ``custom_rush_points`` and configurations inferred from
        ``transfer_learning_evaluations``. These points are not used to prune
        any configurations.
    :param custom_rush_points: If given, these configurations are evaluated
        first, in addition to top performing configurations from other tasks
        and also serve to preemptively prune underperforming configurations
    :param num_hyperparameters_per_task: The number of top hyperparameter
        configurations to consider per task. Defaults to 1
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        transfer_learning_evaluations: Dict[str, LegacyTransferLearningTaskEvaluations],
        metric: str,
        type: str = "stopping",
        points_to_evaluate: Optional[List[dict]] = None,
        custom_rush_points: Optional[List[dict]] = None,
        num_hyperparameters_per_task: int = 1,
        **kwargs,
    ):
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
            type="rush_" + type,
            points_to_evaluate=points_to_evaluate,
            metric_names=[metric],
            rung_system_kwargs={"num_threshold_candidates": num_threshold_candidates},
            **kwargs,
        )
