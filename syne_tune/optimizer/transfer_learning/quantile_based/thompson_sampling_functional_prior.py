from typing import Dict
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split

from blackbox_repository import load
from blackbox_repository.blackbox_surrogate import BlackboxSurrogate
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.searchers import BaseSearcher
from syne_tune.optimizer.transfer_learning import TransferLearningTaskEvaluations
from syne_tune.optimizer.transfer_learning.quantile_based.normalization_transforms import from_string
import pandas as pd

from syne_tune.util import catchtime


def extract_input_output(transfer_learning_evaluations, normalization: str):
    X = pd.concat(
        [evals.hyperparameters for evals in transfer_learning_evaluations.values()],
        ignore_index=True
    )
    normalizer = from_string(normalization)
    y = np.concatenate([
        normalizer(evals.metrics).transform(evals.metrics)
        for evals in transfer_learning_evaluations.values()
    ], axis=0)
    return X, y


def fit_model(config_space, transfer_learning_evaluations, normalization: str, max_fit_samples: int, model=xgboost.XGBRegressor()):
    model_pipeline = BlackboxSurrogate.make_model_pipeline(
        configuration_space=config_space,
        fidelity_space={},
        model=model,
    )
    X, y = extract_input_output(transfer_learning_evaluations, normalization)
    with catchtime("time to fit the model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        X_train, y_train = subsample(X_train, y_train, max_samples=max_fit_samples)
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


def subsample(X_train, z_train, max_samples: int = 10000):
    assert len(X_train) == len(z_train)
    X_train.reset_index(inplace=True)
    if max_samples is not None and max_samples < len(X_train):
        random_indices = np.random.permutation(len(X_train))[:max_samples]
        X_train = X_train.loc[random_indices]
        z_train = z_train[random_indices]
    return X_train, z_train


class TS(BaseSearcher):

    def __init__(
            self,
            config_space: Dict,
            mode: str,
            metric: str,
            transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
            max_fit_samples: int = 100000,
            normalization: str = "gaussian",
    ):
        super(TS, self).__init__(configspace=config_space, metric=metric)
        # todo seed management
        self.mode = mode
        self.model_pipeline, sigma_train, sigma_val = fit_model(
            config_space=config_space,
            transfer_learning_evaluations=transfer_learning_evaluations,
            normalization=normalization,
            max_fit_samples=max_fit_samples,
            model=xgboost.XGBRegressor(),
        )
        print(f"residual train: {sigma_train}")
        print(f"residual val: {sigma_val}")

        with catchtime("time to predict"):
            num_candidates = 10000
            self.X_candidates = pd.DataFrame([self._sample() for _ in range(num_candidates)])
            self.mu_pred = self.model_pipeline.predict(self.X_candidates)
            # simple for now
            if self.mu_pred.ndim == 1:
                self.mu_pred = self.mu_pred.reshape(-1, 1)
            self.sigma_pred = np.ones_like(self.mu_pred) * sigma_val

    def _update(self, trial_id: str, config: Dict, result: Dict):
        pass

    def clone_from_state(self, state):
        pass

    def get_config(self, **kwargs):
        samples = np.random.normal(loc=self.mu_pred, scale=self.sigma_pred)
        if self.mode == 'max':
            samples *= -1
        candidate = self.X_candidates.loc[np.argmin(samples)]
        return dict(candidate)

    def _sample(self):
        return {
            k: v.sample() if hasattr(v, "sample") else v
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
            max_fit_samples=5000,
        ),
        mode="min",
        metric=bb_dict[test_task].objectives_names[metric_index],
        max_t=200,
        resource_attr='hp_epoch',
    )


if __name__ == '__main__':
    run_ts()
