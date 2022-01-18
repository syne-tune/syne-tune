import logging
from typing import Optional, List, Tuple, Dict
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from blackbox_repository import load
from blackbox_repository.blackbox_surrogate import BlackboxSurrogate
from syne_tune.optimizer.schedulers.searchers import BaseSearcher
from syne_tune.optimizer.transfer_learning import TransferLearningTaskEvaluations
from syne_tune.optimizer.transfer_learning.quantile_based.normalization_transforms import from_string
from syne_tune.optimizer.transfer_learning.quantile_based.prior.xgboost_prior import XGBoostPrior
import pandas as pd

class TS(BaseSearcher):

    def __init__(
            self,
            config_space: Dict,
            mode: str,
            metric: str,
            transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
            normalization: str = "standard",
            prior: str = "xgboost",
    ):
        # TODO option to pick only last fidelity
        self.model_pipeline = BlackboxSurrogate.make_model_pipeline(
            configuration_space=config_space,
            fidelity_space={},
            model=KNeighborsRegressor(n_neighbors=1),
        )
        X_train = pd.concat([evals.hyperparameters for evals in transfer_learning_evaluations.values()])
        normalizer = from_string(normalization)
        z_train = np.concatenate([
            normalizer(evals.metrics).transform(evals.metrics)
            for evals in transfer_learning_evaluations.values()
        ], axis=0)
        self.model_pipeline.fit(X_train, z_train)

    def _update(self, trial_id: str, config: Dict, result: Dict):
        pass

    def clone_from_state(self, state):
        pass

    def _sample(self, candidates: Optional[np.array] = None) -> np.array:
        if candidates is None:
            num_random_candidates = 10000
            # since Thompson Sampling selects from discrete set of options,
            # when no candidates are given we draw random candidates
            candidates = self.draw_random_candidates(num_random_candidates)

        mu_pred, sigma_pred = self.prior.predict(candidates)
        samples = np.random.normal(loc=mu_pred, scale=sigma_pred)
        return candidates[np.argmin(samples)]

    def get_config(self, **kwargs):
        # TODO sample N candidates
        # TODO sample TS values
        # TODO return first
        n = 100
        candidates = pd.DataFrame([
            self._sample()
            _ for _ in range(n)
        ])
        return {}

    def _sample(self):
        return {
            k: v.sample() if hasattr(v, "sample") else v
            for k, v in self.config_space.items()
        }



if __name__ == '__main__':

    bb_dict = load("nasbench201")

    test_task = "cifar100"
    config_space = bb_dict[test_task].configuration_space
    metric_index = 0
    transfer_learning_evaluations = {
        task: TransferLearningTaskEvaluations(
            hyperparameters=bb.hyperparameters,
            # average over seed, take last fidelity and pick only first metric
            metrics=bb.objectives_evaluations.mean(axis=1)[:, -1, metric_index:metric_index+1]
        )
        for task, bb in bb_dict.items()
        if task != test_task
    }

    bb_sch = TS(
        mode="min",
        config_space=config_space,
        metric=bb_dict[test_task].objectives_names[metric_index],
        transfer_learning_evaluations=transfer_learning_evaluations,
    )

    print(bb_sch.get_config())