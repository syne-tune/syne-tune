import logging
from typing import Dict, Optional, Any, Tuple
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split

from syne_tune.blackbox_repository.blackbox_surrogate import BlackboxSurrogate

import pandas as pd

from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import (
    SingleObjectiveBaseSearcher,
)
from syne_tune.optimizer.schedulers.transfer_learning.transfer_learning_task_evaluation import (
    TransferLearningTaskEvaluations,
)
from syne_tune.optimizer.schedulers.transfer_learning.quantile_based.normalization_transforms import (
    from_string,
)
from syne_tune.config_space import Domain
from syne_tune.util import catchtime

logger = logging.getLogger(__name__)


def extract_input_output(
    transfer_learning_evaluations, normalization: str, random_state
):
    X = pd.concat(
        [evals.hyperparameters for evals in transfer_learning_evaluations.values()],
        ignore_index=True,
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
    model=xgboost.XGBRegressor(),
):
    model_pipeline = BlackboxSurrogate.make_model_pipeline(
        configuration_space=config_space,
        fidelity_space={},
        model=model,
    )
    X, y = extract_input_output(
        transfer_learning_evaluations, normalization, random_state=random_state
    )
    with catchtime("time to fit the model"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=random_state
        )
        X_train, y_train = subsample(
            X_train, y_train, max_samples=max_fit_samples, random_state=random_state
        )
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


def subsample(
    X: pd.DataFrame,
    y: np.array,
    max_samples: Optional[int] = 10000,
    random_state: np.random.RandomState = None,
) -> Tuple[pd.DataFrame, np.array]:
    """
    Subsample both X and y with `max_samples` elements. If `max_samples` is not set then X and y are returned as such
    and if it is set, the index of X is reset.
    :return: (X, y) with `max_samples` sampled elements.
    """
    assert len(X) == len(y)
    if max_samples is not None and max_samples < len(X):
        if random_state is None:
            random_indices = np.random.permutation(len(X))[:max_samples]
        else:
            random_indices = random_state.permutation(len(X))[:max_samples]
        # reset the index to be able to address elements between [0, len(X_train)-1]
        X.reset_index(inplace=True, drop=True)
        X = X.loc[random_indices]
        y = y[random_indices]
    return X, y


class QuantileBasedSurrogateSearcher(SingleObjectiveBaseSearcher):
    """
    Implements the transfer-learning method:

        | A Quantile-based Approach for Hyperparameter Transfer Learning.
        | David Salinas, Huibin Shen, Valerio Perrone.
        | ICML 2020.

    This is the Copula Thompson Sampling approach described in the paper where a
    surrogate is fitted on the transfer learning data to predict mean/variance of
    configuration performance given a hyperparameter. The surrogate is then sampled
    from and the best configurations are returned as next candidate to evaluate.

    :param config_space: Configuration space for the evaluation function.
    :param transfer_learning_evaluations: Dictionary from task name to offline
        evaluations.
    :param max_fit_samples: Maximum number to use when fitting the method.
        Defaults to 100000
    :param normalization: Default to "gaussian" which first computes the rank
        and then applies Gaussian inverse CDF. "standard" applies just
        standard normalization (remove mean and divide by variance) but can
        perform significantly worse.
    :param random_seed: Seed for the random number generator.
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
        max_fit_samples: int = 100000,
        normalization: str = "gaussian",
        random_seed: int = None,
    ):
        super(QuantileBasedSurrogateSearcher, self).__init__(
            config_space=config_space, points_to_evaluate=[], random_seed=random_seed
        )

        self.random_state = np.random.RandomState(self.random_seed)
        self.model_pipeline, sigma_train, sigma_val = fit_model(
            config_space=config_space,
            transfer_learning_evaluations=transfer_learning_evaluations,
            normalization=normalization,
            max_fit_samples=max_fit_samples,
            model=xgboost.XGBRegressor(),
            random_state=self.random_state,
        )
        logger.info(f"residual train: {sigma_train}\nresidual val: {sigma_val}")

        with catchtime("time to predict"):
            # note the candidates could also be sampled every time, we cache them rather to save compute time.
            num_candidates = 100000
            self.X_candidates = pd.DataFrame(
                [self._sample_random_config() for _ in range(num_candidates)]
            )
            self.mu_pred = self.model_pipeline.predict(self.X_candidates)
            # simple homoskedastic variance estimate for now
            if self.mu_pred.ndim == 1:
                self.mu_pred = self.mu_pred.reshape(-1, 1)
            self.sigma_pred = np.ones_like(self.mu_pred) * sigma_val

    def _update(self, trial_id: str, config: Dict[str, Any], result: Dict[str, Any]):
        pass

    def clone_from_state(self, state: Dict[str, Any]):
        raise NotImplementedError

    def suggest(self, **kwargs) -> Optional[Dict[str, Any]]:
        samples = self.random_state.normal(loc=self.mu_pred, scale=self.sigma_pred)
        candidate = self.X_candidates.loc[np.argmin(samples)]
        return dict(candidate)

    def _sample_random_config(self):
        return {
            k: v.sample(random_state=self.random_state) if isinstance(v, Domain) else v
            for k, v in self.config_space.items()
        }
