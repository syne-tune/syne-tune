import logging
from typing import Dict, Optional, Any

import pandas as pd
import xgboost

from syne_tune.blackbox_repository.blackbox_surrogate import BlackboxSurrogate
from syne_tune.config_space import Domain
from syne_tune.optimizer.schedulers.searchers import StochasticSearcher
from syne_tune.optimizer.schedulers.transfer_learning import (
    LegacyTransferLearningTaskEvaluations,
    LegacyTransferLearningMixin,
)

logger = logging.getLogger(__name__)


class LegacyZeroShotTransfer(LegacyTransferLearningMixin, StochasticSearcher):
    """
    A zero-shot transfer hyperparameter optimization method which jointly selects
    configurations that minimize the average rank obtained on historic metadata
    (``transfer_learning_evaluations``). This is a searcher which can be used
    with :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`. Reference:

        | Sequential Model-Free Hyperparameter Tuning.
        | Martin Wistuba, Nicolas Schilling, Lars Schmidt-Thieme.
        | IEEE International Conference on Data Mining (ICDM) 2015.

    Additional arguments on top of parent class
    :class:`~syne_tune.optimizer.schedulers.searchers.StochasticSearcher`:

    :param transfer_learning_evaluations: Dictionary from task name to
        offline evaluations.
    :param mode: Whether to minimize ("min", default) or maximize ("max")
    :param sort_transfer_learning_evaluations: Use ``False`` if the
        hyperparameters for each task in ``transfer_learning_evaluations`` are
        already in the same order. If set to ``True``, hyperparameters are sorted.
        Defaults to ``True``
    :param use_surrogates: If the same configuration is not evaluated on all
        tasks, set this to ``True``. This will generate a set of configurations
        and will impute their performance using surrogate models.
        Defaults to ``False``
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        transfer_learning_evaluations: Dict[str, LegacyTransferLearningTaskEvaluations],
        mode: str = "min",
        sort_transfer_learning_evaluations: bool = True,
        use_surrogates: bool = False,
        **kwargs,
    ) -> None:
        if "points_to_evaluate" in kwargs:
            kwargs["points_to_evaluate"] = []
        super().__init__(
            config_space=config_space,
            metric=metric,
            transfer_learning_evaluations=transfer_learning_evaluations,
            metric_names=[metric],
            **kwargs,
        )
        self._mode = mode
        if use_surrogates and len(transfer_learning_evaluations) <= 1:
            use_surrogates = False
            sort_transfer_learning_evaluations = False
        if use_surrogates:
            sort_transfer_learning_evaluations = False
            transfer_learning_evaluations = (
                self._create_surrogate_transfer_learning_evaluations(
                    config_space, transfer_learning_evaluations, metric
                )
            )
        warning_message = "This searcher assumes that each hyperparameter configuration occurs in all tasks. "
        scores = list()
        hyperparameters = None
        for task_name, task_data in transfer_learning_evaluations.items():
            assert (
                hyperparameters is None
                or task_data.hyperparameters.shape == hyperparameters.shape
            ), warning_message
            hyperparameters = task_data.hyperparameters
            if sort_transfer_learning_evaluations:
                hyperparameters = task_data.hyperparameters.sort_values(
                    list(task_data.hyperparameters.columns)
                )
            idx = hyperparameters.index.values
            avg_scores = task_data.objective_values(metric).mean(axis=1)
            if self._mode == "max":
                avg_scores = avg_scores.max(axis=1)[idx]
            else:
                avg_scores = avg_scores.min(axis=1)[idx]
            scores.append(avg_scores)
        if not use_surrogates:
            if len(transfer_learning_evaluations) > 1:
                logger.warning(
                    warning_message
                    + "If this is not the case, this searcher fails without a warning."
                )
            if not sort_transfer_learning_evaluations:
                hyperparameters = hyperparameters.copy()
        hyperparameters.reset_index(drop=True, inplace=True)
        self._hyperparameters = hyperparameters
        sign = 1 if self._mode == "min" else -1
        self._scores = sign * pd.DataFrame(scores)
        self._ranks = self._update_ranks()

    def _create_surrogate_transfer_learning_evaluations(
        self,
        config_space: Dict[str, Any],
        transfer_learning_evaluations: Dict[str, LegacyTransferLearningTaskEvaluations],
        metric: str,
    ) -> Dict[str, LegacyTransferLearningTaskEvaluations]:
        """
        Creates transfer_learning_evaluations where each configuration is evaluated on each task using surrogate models.
        """
        surrogate_transfer_learning_evaluations = dict()
        for task_name, task_data in transfer_learning_evaluations.items():
            estimator = BlackboxSurrogate.make_model_pipeline(
                configuration_space=config_space,
                fidelity_space={},
                model=xgboost.XGBRegressor(),
            )
            X_train = task_data.hyperparameters
            y_train = task_data.objective_values(metric).mean(axis=1)
            if self._mode == "max":
                y_train = y_train.max(axis=1)
            else:
                y_train = y_train.min(axis=1)
            estimator.fit(X_train, y_train)

            num_candidates = 10000 if len(config_space) >= 6 else 5 ** len(config_space)
            hyperparameters_new = pd.DataFrame(
                [
                    self._sample_random_config(config_space)
                    for _ in range(num_candidates)
                ]
            )
            objectives_evaluations_new = estimator.predict(hyperparameters_new).reshape(
                -1, 1, 1, 1
            )
            surrogate_transfer_learning_evaluations[
                task_name
            ] = LegacyTransferLearningTaskEvaluations(
                configuration_space=config_space,
                hyperparameters=hyperparameters_new,
                objectives_names=[metric],
                objectives_evaluations=objectives_evaluations_new,
            )
        return surrogate_transfer_learning_evaluations

    def get_config(self, **kwargs) -> Optional[dict]:
        if self._ranks.shape[1] == 0:
            return None
        # Select greedy-best configuration considering all others
        best_idx = self._ranks.mean(axis=0).idxmin()
        # Update ranks for choosing each configuration considering the previously chosen ones
        self._ranks.clip(upper=self._ranks[best_idx], axis=0, inplace=True)
        # Drop the chosen configuration as a future candidate
        self._scores.drop(columns=best_idx, inplace=True)
        best_config = self._hyperparameters.loc[best_idx]
        self._hyperparameters.drop(index=best_idx, inplace=True)
        if self._ranks.std(axis=1).sum() == 0:
            self._ranks = self._update_ranks()
        return best_config.to_dict()

    def _sample_random_config(self, config_space: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v.sample(random_state=self.random_state) if isinstance(v, Domain) else v
            for k, v in config_space.items()
        }

    def _update_ranks(self) -> pd.DataFrame:
        return self._scores.rank(axis=1)

    def _update(
        self, trial_id: str, config: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        pass
