import numpy as np
import pytest

from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.quantile_regression_model import (
    QuantileRegressor,
    QuantileRegressorPredictions,
)
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.symmetric_conformalized_quantile_regression_model import (
    SymmetricConformalizedGradientBoostingQuantileRegressor,
    SymmetricConformalizedQuantileRegressor,
    apply_symmetric_conformal_corrections,
    symmetric_conformal_corrections,
)

quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]


def constant_predictions(n: int, value: float = 0.0) -> QuantileRegressorPredictions:
    return QuantileRegressorPredictions(
        quantiles=quantiles, results_stacked=np.full((n, len(quantiles)), value)
    )


def test_corrections_reach_target_coverage():
    y = np.random.RandomState(0).normal(size=200)
    # predicting zero for every quantile undercovers, corrections must widen bands
    predictions = constant_predictions(len(y))
    corrections = symmetric_conformal_corrections(predictions, y)
    assert set(corrections.keys()) == {0.1, 0.3}
    assert corrections[0.1] > corrections[0.3] > 0

    corrected = apply_symmetric_conformal_corrections(predictions, corrections)
    for lower, upper in [(0.1, 0.9), (0.3, 0.7)]:
        covered = (corrected.results(lower) <= y) & (y <= corrected.results(upper))
        assert covered.mean() >= 1 - 2 * lower
    # median is not corrected
    assert np.all(corrected.results(0.5) == predictions.results(0.5))


class ConstantQuantileRegressor(QuantileRegressor):
    def __init__(self):
        self.quantiles = quantiles
        self.num_fit_samples = None

    def fit(self, df_features, y, **kwargs):
        self.num_fit_samples = len(y)

    def predict(self, df_test) -> QuantileRegressorPredictions:
        return constant_predictions(len(df_test))


@pytest.mark.parametrize("num_samples, conformalized", [(10, False), (100, True)])
def test_conformalized_quantile_regressor(num_samples, conformalized):
    random_state = np.random.RandomState(0)
    X = random_state.uniform(size=(num_samples, 2))
    y = random_state.normal(size=num_samples)
    regressor = SymmetricConformalizedQuantileRegressor(
        base_regressor=ConstantQuantileRegressor(),
        valid_fraction=0.2,
        min_samples_to_conformalize=32,
    )
    regressor.fit(X, y)
    predictions = regressor.predict(X)
    assert predictions.results_stacked.shape == (num_samples, len(quantiles))
    if conformalized:
        assert regressor.base_regressor.num_fit_samples == 80
        assert predictions.results(0.1)[0] < 0 < predictions.results(0.9)[0]
    else:
        assert regressor.base_regressor.num_fit_samples == num_samples
        assert np.all(predictions.results_stacked == 0)


@pytest.mark.timeout(10)
def test_conformalized_gradient_boosting():
    random_state = np.random.RandomState(0)
    X = random_state.uniform(size=(100, 2))
    y = X.sum(axis=1) + random_state.normal(scale=0.1, size=100)
    regressor = SymmetricConformalizedGradientBoostingQuantileRegressor(
        quantiles=5, min_samples_to_conformalize=32
    )
    regressor.fit(X, y)
    predictions = regressor.predict(X)
    assert predictions.results_stacked.shape == (100, 5)
