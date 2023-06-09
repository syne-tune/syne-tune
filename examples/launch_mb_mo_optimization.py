# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import copy
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.linear_model import BayesianRidge

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint, uniform
from syne_tune.optimizer.schedulers import FIFOScheduler
from syne_tune.optimizer.schedulers.multiobjective.multi_surrogate_multi_objective_searcher import (
    MultiObjectiveMultiSurrogateSearcher,
)
from syne_tune.optimizer.schedulers.multiobjective.random_scalarization import (
    MultiObjectiveLCBRandomLinearScalarization,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.sklearn_model import (
    SKLearnEstimatorWrapper,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn import (
    SKLearnEstimator,
    SKLearnPredictor,
)


class BayesianRidgePredictor(SKLearnPredictor):
    """
    Base class for the sklearn predictors
    """

    def __init__(self, ridge: BayesianRidge):
        self.ridge = ridge

    def predict(
        self, X: np.ndarray, return_std: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns signals which are statistics of the predictive distribution at
        input points ``inputs``.


        :param inputs: Input points, shape ``(n, d)``
        :return: Tuple with the following entries:
            * "mean": Predictive means in shape of ``(n,)``
            * "std": Predictive stddevs, shape ``(n,)``
        """
        return self.ridge.predict(X, return_std=True)


class BayesianRidgeEstimator(SKLearnEstimator):
    """
    Base class for the sklearn Estimators
    """

    def __init__(self, *args, **kwargs):
        self.ridge = BayesianRidge(*args, **kwargs)

    def fit(
        self, X: np.ndarray, y: np.ndarray, update_params: bool
    ) -> BayesianRidgePredictor:
        """
        Implements :meth:`fit_from_state`, given transformed data.

        :param X: Training data in ndarray of shape (n_samples, n_features)
        :param y: Target values in ndarray of shape (n_samples,)
        :param update_params: Should model (hyper)parameters be updated?
        :return: Predictor, wrapping the posterior state
        """
        self.ridge.fit(X, y.ravel())
        return BayesianRidgePredictor(ridge=copy.deepcopy(self.ridge))


def main():
    # Hyperparameter configuration space
    config_space = {
        "steps": randint(0, 100),
        "theta": uniform(0, np.pi / 2),
        "sleep_time": 0.01,
    }
    # Scheduler (i.e., HPO algorithm)
    sklearn_myestimators = {
        "y1": BayesianRidgeEstimator(),
        "y2": BayesianRidgeEstimator(),
    }
    myestimators = {
        metric: SKLearnEstimatorWrapper(estim, target_metric=metric)
        for metric, estim in sklearn_myestimators.items()
    }
    searcher = MultiObjectiveMultiSurrogateSearcher(
        config_space=config_space,
        metric=["y1", "y2"],
        estimators=myestimators,
        scoring_class_and_args=MultiObjectiveLCBRandomLinearScalarization,
    )

    scheduler = FIFOScheduler(
        config_space,
        metric=["y1", "y2"],
        mode=["min", "min"],
        searcher=searcher,
        search_options={"debug_log": False},
    )

    entry_point = str(
        Path(__file__).parent
        / "training_scripts"
        / "mo_artificial"
        / "mo_artificial.py"
    )
    tuner = Tuner(
        trial_backend=LocalBackend(entry_point=entry_point),
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_wallclock_time=30),
        n_workers=1,  # how many trials are evaluated in parallel
    )
    tuner.run()


if __name__ == "__main__":
    main()
