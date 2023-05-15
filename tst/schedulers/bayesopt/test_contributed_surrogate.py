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
from typing import Tuple

import numpy as np
import pytest

from syne_tune.config_space import uniform, choice
from syne_tune.optimizer.schedulers.searchers.bayesopt.contributed.estimator import (
    ContributedEstimator,
    ContributedEstimatorWrapper,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.contributed.predictor import (
    ContributedPredictor,
    ContributedPredictorWrapper,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    dictionarize_objective,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.test_objects import (
    create_tuning_job_state,
)
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges_factory import (
    make_hyperparameter_ranges,
)


class TestPredictor(ContributedPredictor):
    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        nexamples = X.shape[0]
        return np.ones_like(nexamples), np.zeros(nexamples)


class TestEstimator(ContributedEstimator):
    def fit(self, X: np.ndarray, y: np.ndarray, update_params: bool) -> TestPredictor:
        # Assert the right data is passed to the fit
        np.testing.assert_allclose(X[:, 0], np.array([0.2, 0.31, 0.15]))
        np.testing.assert_allclose(y.ravel(), np.array([1.0, 2.0, 0.3]))
        return TestPredictor()


@pytest.fixture
def tuning_job_state() -> TuningJobState:
    hp_ranges1 = make_hyperparameter_ranges(
        {"a1_hp_1": uniform(-5.0, 5.0), "a1_hp_2": choice(["a", "b", "c"])}
    )
    X1 = [(-3.0, "a"), (-1.9, "c"), (-3.5, "a")]
    Y1 = [dictionarize_objective(y) for y in (1.0, 2.0, 0.3)]

    return create_tuning_job_state(hp_ranges=hp_ranges1, cand_tuples=X1, metrics=Y1)


def test_estimator_wrapper_interface(tuning_job_state):
    estimator = ContributedEstimatorWrapper(contributed_estimator=TestEstimator())
    predictor = estimator.fit_from_state(tuning_job_state)

    assert isinstance(predictor, ContributedPredictorWrapper)
    assert isinstance(predictor.contributed_predictor, TestPredictor)
    assert isinstance(estimator, ContributedEstimatorWrapper)
    assert isinstance(estimator.contributed_estimator, TestEstimator)


def test_predictor_wrapper_interface(tuning_job_state):
    estimator = ContributedEstimatorWrapper(contributed_estimator=TestEstimator())
    predictor = estimator.fit_from_state(tuning_job_state)
    predictions = predictor.predict(np.random.uniform(size=(10, 3)))

    np.testing.assert_allclose(predictions[0]["mean"], np.ones(shape=10))
    np.testing.assert_allclose(predictions[0]["std"], np.zeros(shape=10))
