import logging
from typing import Optional, List, Tuple, Dict
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from blackbox_repository import load
from blackbox_repository.blackbox_surrogate import BlackboxSurrogate
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.searchers import BaseSearcher
from syne_tune.optimizer.transfer_learning import TransferLearningTaskEvaluations
from syne_tune.optimizer.transfer_learning.quantile_based.normalization_transforms import from_string
from syne_tune.optimizer.transfer_learning.quantile_based.prior.xgboost_prior import XGBoostPrior
import pandas as pd

from syne_tune.util import catchtime


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
        super(TS, self).__init__(configspace=config_space, metric=metric)
        self.mode = mode
        # TODO option to pick only last fidelity

        self.model_pipeline = BlackboxSurrogate.make_model_pipeline(
            configuration_space=config_space,
            fidelity_space={},
            model=KNeighborsRegressor(n_neighbors=3),
        )
        df_train = pd.concat([evals.hyperparameters for evals in transfer_learning_evaluations.values()])
        normalizer = from_string(normalization)
        z_train = np.concatenate([
            normalizer(evals.metrics).transform(evals.metrics)
            for evals in transfer_learning_evaluations.values()
        ], axis=0)
        with catchtime("time to fit the model"):
            self.model_pipeline.fit(df_train, z_train)
            # (num_hps, num_metrics)
            mu_pred_train = self.model_pipeline.predict(df_train)
            # compute residuals (num_metrics,)
            sigma_train = np.std(z_train - mu_pred_train, axis=0)

        with catchtime("time to predict"):
            num_candidates = 10000
            self.df_candidates = pd.DataFrame([self._sample() for _ in range(num_candidates)])
            self.mu_pred = self.model_pipeline.predict(self.df_candidates)
            # simple for now
            self.sigma_pred = np.ones_like(self.mu_pred) * sigma_train

    def _update(self, trial_id: str, config: Dict, result: Dict):
        pass

    def clone_from_state(self, state):
        pass

    def get_config(self, **kwargs):
        samples = np.random.normal(loc=self.mu_pred, scale=self.sigma_pred)
        if self.mode == 'max':
            samples *= -1
        candidate = self.df_candidates.loc[np.argmin(samples)]
        return dict(candidate)

    def _sample(self):
        return {
            k: v.sample() if hasattr(v, "sample") else v
            for k, v in self.configspace.items()
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
            metrics=bb.objectives_evaluations.mean(axis=1)[:, -1, metric_index:metric_index + 1]
        )
        for task, bb in bb_dict.items()
        if task != test_task
    }


    blackbox = bb_dict[test_task]
    HyperbandScheduler(
        config_space=config_space,
        searcher=TS(
            mode="min",
            config_space=config_space,
            metric=bb_dict[test_task].objectives_names[metric_index],
            transfer_learning_evaluations=transfer_learning_evaluations,
        ),
        mode="min",
        metric=bb_dict[test_task].objectives_names[metric_index],
        max_t=200,
        resource_attr='hp_epoch',
    )
#
# if __name__ == '__main__':
#
#     bb_dict = load("nasbench201")
#
#     test_task = "cifar100"
#     config_space = bb_dict[test_task].configuration_space
#     metric_index = 0
#     transfer_learning_evaluations = {
#         task: TransferLearningTaskEvaluations(
#             hyperparameters=bb.hyperparameters,
#             # average over seed, take last fidelity and pick only first metric
#             metrics=bb.objectives_evaluations.mean(axis=1)[:, -1, metric_index:metric_index+1]
#         )
#         for task, bb in bb_dict.items()
#         if task != test_task
#     }
#
#     bb_sch = BoundingBox(
#         scheduler_fun=lambda new_config_space, mode, metric: FIFOScheduler(
#             new_config_space,
#             points_to_evaluate=[],
#             searcher='random',
#             metric=metric,
#             mode=mode,
#         ),
#         mode="min",
#         config_space=config_space,
#         metric=bb_dict[test_task].objectives_names[metric_index],
#         transfer_learning_evaluations=transfer_learning_evaluations,
#     )
