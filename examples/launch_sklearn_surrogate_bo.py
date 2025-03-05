import copy
from pathlib import Path
from typing import Tuple
import logging

import numpy as np
from sklearn.linear_model import BayesianRidge

from examples.training_scripts.height_example.train_height import (
    METRIC_ATTR,
    METRIC_MODE,
    MAX_RESOURCE_ATTR,
)
from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint
from syne_tune.optimizer.schedulers import LegacyFIFOScheduler
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc_impl import (
    EIAcquisitionFunction,
)
from syne_tune.optimizer.schedulers.searchers.sklearn import (
    SKLearnSurrogateSearcher,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn import (
    SKLearnEstimator,
    SKLearnPredictor,
)


class BayesianRidgePredictor(SKLearnPredictor):
    """
    Predictor for surrogate model given by ``sklearn.linear_model.BayesianRidge``.
    """

    def __init__(self, ridge: BayesianRidge):
        self.ridge = ridge

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.ridge.predict(X, return_std=True)


class BayesianRidgeEstimator(SKLearnEstimator):
    """
    Estimator for surrogate model given by ``sklearn.linear_model.BayesianRidge``.

    None of the parameters of ``BayesianRidge`` are exposed here, so they are all
    fixed up front.
    """

    def __init__(self, *args, **kwargs):
        self.ridge = BayesianRidge(*args, **kwargs)

    def fit(
        self, X: np.ndarray, y: np.ndarray, update_params: bool
    ) -> SKLearnPredictor:
        self.ridge.fit(X, y.ravel())
        return BayesianRidgePredictor(ridge=copy.deepcopy(self.ridge))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    random_seed = 31415927
    max_epochs = 100
    n_workers = 4

    config_space = {
        "width": randint(1, 20),
        "height": randint(1, 20),
        MAX_RESOURCE_ATTR: 100,
    }
    entry_point = str(
        Path(__file__).parent
        / "training_scripts"
        / "height_example"
        / "train_height.py"
    )

    # We use ``FIFOScheduler`` with a specific searcher based on our surrogate
    # model
    searcher = SKLearnSurrogateSearcher(
        config_space=config_space,
        metric=METRIC_ATTR,
        estimator=BayesianRidgeEstimator(),
        scoring_class=EIAcquisitionFunction,
    )
    scheduler = LegacyFIFOScheduler(
        config_space,
        metric=METRIC_ATTR,
        mode=METRIC_MODE,
        max_resource_attr=MAX_RESOURCE_ATTR,
        searcher=searcher,
    )

    tuner = Tuner(
        trial_backend=LocalBackend(entry_point=entry_point),
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_wallclock_time=60),
        n_workers=n_workers,
    )

    tuner.run()
