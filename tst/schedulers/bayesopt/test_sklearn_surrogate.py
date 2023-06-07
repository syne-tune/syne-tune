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
from syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn.estimator import (
    SKLearnEstimator,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn.predictor import (
    SKLearnPredictor,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.sklearn_model import (
    SKLearnEstimatorWrapper,
    SKLearnPredictorWrapper,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    dictionarize_objective,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.test_objects import (
    create_tuning_job_state,
    tuples_to_configs,
)
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges_factory import (
    make_hyperparameter_ranges,
)


class TestPredictor(SKLearnPredictor):
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nexamples = X.shape[0]
        return np.ones_like(nexamples), np.zeros(nexamples)


class TestEstimator(SKLearnEstimator):
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


def test_estimator_wrapper_interface(tuning_job_state: TuningJobState):
    estimator = SKLearnEstimatorWrapper(sklearn_estimator=TestEstimator())
    predictor = estimator.fit_from_state(tuning_job_state, update_params=False)

    assert isinstance(predictor, SKLearnPredictorWrapper)
    assert isinstance(predictor.sklearn_predictor, TestPredictor)
    assert isinstance(estimator, SKLearnEstimatorWrapper)
    assert isinstance(estimator.sklearn_estimator, TestEstimator)


def test_predictor_wrapper_interface(tuning_job_state: TuningJobState):
    estimator = SKLearnEstimatorWrapper(sklearn_estimator=TestEstimator())
    predictor = estimator.fit_from_state(tuning_job_state, update_params=False)
    predictions = predictor.predict(np.random.uniform(size=(10, 3)))

    np.testing.assert_allclose(predictions[0]["mean"], np.ones(shape=10))
    np.testing.assert_allclose(predictions[0]["std"], np.zeros(shape=10))


def test_pending_evaluations(tuning_job_state: TuningJobState):
    pending = tuples_to_configs(
        [(1.0, "a")],
        make_hyperparameter_ranges(
            {"a1_hp_1": uniform(-5.0, 5.0), "a1_hp_2": choice(["a", "b", "c"])}
        ),
    )
    tuning_job_state.append_pending("123", pending.pop())
    estimator = SKLearnEstimatorWrapper(sklearn_estimator=TestEstimator())
    predictor = estimator.fit_from_state(tuning_job_state, update_params=False)

    assert isinstance(predictor, SKLearnPredictorWrapper)
    assert isinstance(predictor.sklearn_predictor, TestPredictor)
