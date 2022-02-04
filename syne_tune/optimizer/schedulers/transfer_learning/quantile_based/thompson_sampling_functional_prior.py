from typing import Dict, Optional
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split

from benchmarking.blackbox_repository.blackbox_surrogate import BlackboxSurrogate
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.searchers import BaseSearcher, SearcherWithRandomSeed

import pandas as pd

from syne_tune.optimizer.schedulers.transfer_learning import TransferLearningTaskEvaluations
from syne_tune.optimizer.schedulers.transfer_learning.quantile_based.normalization_transforms import from_string
from syne_tune.search_space import Domain
from syne_tune.util import catchtime


def extract_input_output(transfer_learning_evaluations, normalization: str, random_state):
    X = pd.concat(
        [evals.hyperparameters for evals in transfer_learning_evaluations.values()],
        ignore_index=True
    )
    normalizer = from_string(normalization)
    ys = []
    for evals in transfer_learning_evaluations.values():
        # take average over seed and last fidelity and first objective
        y = evals.objectives_evaluations.mean(axis=1)[:, -1, 0:1]
        ys.append(normalizer(y, random_state=random_state).transform(y))
    y = np.concatenate(ys, axis=0)
    return X, y


def fit_model(
        config_space,
        transfer_learning_evaluations,
        normalization: str,
        max_fit_samples: int,
        random_state,
        model=xgboost.XGBRegressor()
):
    model_pipeline = BlackboxSurrogate.make_model_pipeline(
        configuration_space=config_space,
        fidelity_space={},
        model=model,
    )
    X, y = extract_input_output(transfer_learning_evaluations, normalization, random_state=random_state)
    with catchtime("time to fit the model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
        X_train, y_train = subsample(X_train, y_train, max_samples=max_fit_samples, random_state=random_state)
        model_pipeline.fit(X_train, y_train)

        # compute residuals (num_metrics,)
        sigma_train = eval_model(model_pipeline, X_train, y_train)

        # compute residuals (num_metrics,)
        sigma_val = eval_model(model_pipeline, X_test, y_test)

    return model_pipeline, sigma_train, sigma_val


def eval_model(model_pipeline, X, y):
    # compute residuals (num_metrics,)
    mu_pred = model_pipeline.predict(X)
    if mu_pred.ndim == 1:
        mu_pred = mu_pred.reshape(-1, 1)
    res = np.std(y - mu_pred, axis=0)
    return res.mean()


def subsample(X_train, z_train, max_samples: int = 10000, random_state: np.random.RandomState = None):
    assert len(X_train) == len(z_train)
    X_train.reset_index(inplace=True)
    if max_samples is not None and max_samples < len(X_train):
        if random_state is None:
            random_indices = np.random.permutation(len(X_train))[:max_samples]
        else:
            random_indices = random_state.permutation(len(X_train))[:max_samples]
        X_train = X_train.loc[random_indices]
        z_train = z_train[random_indices]
    return X_train, z_train


class TS(SearcherWithRandomSeed):

    def __init__(
            self,
            config_space: Dict,
            mode: str,
            metric: str,
            transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
            max_fit_samples: int = 100000,
            normalization: str = "gaussian",
            random_seed: Optional[int] = None,
            **kwargs
    ):
        # todo check points_to_evaluate does not do anything unwanted
        super(TS, self).__init__(
            configspace=config_space,
            metric=metric,
            random_seed=random_seed,
            points_to_evaluate=[],
            **kwargs,
        )
        self.mode = mode
        self.model_pipeline, sigma_train, sigma_val = fit_model(
            config_space=config_space,
            transfer_learning_evaluations=transfer_learning_evaluations,
            normalization=normalization,
            max_fit_samples=max_fit_samples,
            model=xgboost.XGBRegressor(),
            random_state=self.random_state,
        )
        print(f"residual train: {sigma_train}")
        print(f"residual val: {sigma_val}")

        with catchtime("time to predict"):
            num_candidates = 10000
            self.X_candidates = pd.DataFrame([self._sample() for _ in range(num_candidates)])
            self.mu_pred = self.model_pipeline.predict(self.X_candidates)
            # simple variance estimate for now
            if self.mu_pred.ndim == 1:
                self.mu_pred = self.mu_pred.reshape(-1, 1)
            self.sigma_pred = np.ones_like(self.mu_pred) * sigma_val

    def _update(self, trial_id: str, config: Dict, result: Dict):
        pass

    def clone_from_state(self, state):
        pass

    def get_config(self, **kwargs):
        samples = self.random_state.normal(loc=self.mu_pred, scale=self.sigma_pred)
        if self.mode == 'max':
            samples *= -1
        candidate = self.X_candidates.loc[np.argmin(samples)]
        return dict(candidate)

    def _sample(self):
        return {
            k: v.sample(random_state=self.random_state) if isinstance(v, Domain) else v
            for k, v in self.configspace.items()
        }

def run_ts():
    bb, test_task = "nasbench201", "cifar100"
    bb, test_task = "fcnet", "protein_structure"
    bb_dict = load(bb)

    config_space = bb_dict[test_task].configuration_space
    metric_index = 0
    transfer_learning_evaluations = {
        task: TransferLearningTaskEvaluations(
            configuration_space=bb.configuration_space,
            hyperparameters=bb.hyperparameters,
            objectives_evaluations=bb.objectives_evaluations[..., metric_index:metric_index + 1],
            objectives_names=[bb.objectives_names[metric_index]],
        )
        for task, bb in bb_dict.items()
        if task != test_task
    }

    for _ in range(3):

        sch = HyperbandScheduler(
            config_space=config_space,
            searcher=TS(
                mode="min",
                config_space=config_space,
                metric=bb_dict[test_task].objectives_names[metric_index],
                transfer_learning_evaluations=transfer_learning_evaluations,
                max_fit_samples=5000,
                seed=1,
            ),
            mode="min",
            metric=bb_dict[test_task].objectives_names[metric_index],
            max_t=200,
            resource_attr='hp_epoch',
        )
        for i in range(10):
            print(sch.suggest(i))


if __name__ == '__main__':
    from benchmarking.blackbox_repository import load
    run_ts()
