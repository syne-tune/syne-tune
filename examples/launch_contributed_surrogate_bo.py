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
from syne_tune.config_space import randint
from syne_tune.optimizer.schedulers import FIFOScheduler
from syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn.contributed_surrogate_searcher import (
    ContributedSurrogateSearcher,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn.estimator import (
    SklearnEstimator,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn.predictor import (
    SklearnPredictor,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm_components import (
    IndependentThompsonSampling,
)


class BayesianRidgePredictor(SklearnPredictor):
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
        print(f"Predicting with BayesianRidgePredictor using X.shape={X.shape}")
        return self.ridge.predict(X, return_std=True)


class BayesianRidgeEstimator(SklearnEstimator):
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
        print(
            f"Fitting BayesianRidgePredictor using X.shape={X.shape}, y.shape={y.shape}"
        )
        self.ridge.fit(X, y.ravel())
        return BayesianRidgePredictor(ridge=copy.deepcopy(self.ridge))


def main():
    # Hyperparameter configuration space
    config_space = {
        "width": randint(1, 20),
        "height": randint(1, 20),
        "epochs": 10,
    }
    # Scheduler (i.e., HPO algorithm)
    myestimator = BayesianRidgeEstimator()
    searcher = ContributedSurrogateSearcher(
        config_space=config_space,
        metric="mean_loss",
        estimator=myestimator,
        scoring_class_and_args=IndependentThompsonSampling,
    )

    scheduler = FIFOScheduler(
        config_space,
        metric="mean_loss",
        mode="min",
        searcher=searcher,
        search_options={"debug_log": False},
    )

    entry_point = str(
        Path(__file__).parent
        / "training_scripts"
        / "height_example"
        / "train_height_simple.py"
    )
    tuner = Tuner(
        trial_backend=LocalBackend(entry_point=entry_point),
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_wallclock_time=300),
        n_workers=4,  # how many trials are evaluated in parallel
    )
    tuner.run()


if __name__ == "__main__":
    main()
