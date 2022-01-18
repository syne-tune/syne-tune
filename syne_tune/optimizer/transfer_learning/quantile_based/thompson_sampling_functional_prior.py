import logging
from typing import Optional, List, Tuple, Dict
import numpy as np

from syne_tune.optimizer.schedulers.searchers import BaseSearcher
from syne_tune.optimizer.transfer_learning import TransferLearningTaskEvaluations
from syne_tune.optimizer.transfer_learning.quantile_based.normalization_transforms import from_string
from syne_tune.optimizer.transfer_learning.quantile_based.prior.xgboost_prior import XGBoostPrior

"""
class TransferLearningScheduler(TrialScheduler):
    def __init__(
            self,
            config_space: Dict,
            transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
            metric_names: List[str],
    ):

"""
class TS(BaseSearcher):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            bounds: Optional[np.array] = None,
            transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
            normalization: str = "standard",
            prior: str = "xgboost",
    ):
        super(TS, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            evaluations_other_tasks=evaluations_other_tasks,
            bounds=bounds,
        )



        X_train = np.concatenate([X for X, y, _ in evaluations_other_tasks], axis=0)
        normalizer = from_string(normalization)
        z_train = np.concatenate([normalizer(y).transform(y) for X, y, _ in evaluations_other_tasks], axis=0)

        prior_dict = {
            "xgboost": XGBoostPrior,
        }

        logging.info(f"fit prior {prior}")
        self.prior = prior_dict[prior](
            X_train=X_train,
            y_train=z_train,
        )
        logging.info("prior fitted")

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
        return {}

    def draw_random_candidates(self, num_random_candidates: int):
        random_sampler = RS(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            bounds=self.bounds,
        )
        candidates = np.stack([random_sampler.sample() for _ in range(num_random_candidates)])
        return candidates


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

    bb_sch = BoundingBox(
        scheduler_fun=lambda new_config_space, mode, metric: FIFOScheduler(
            new_config_space,
            points_to_evaluate=[],
            searcher='random',
            metric=metric,
            mode=mode,
        ),
        mode="min",
        config_space=config_space,
        metric=bb_dict[test_task].objectives_names[metric_index],
        transfer_learning_evaluations=transfer_learning_evaluations,
    )