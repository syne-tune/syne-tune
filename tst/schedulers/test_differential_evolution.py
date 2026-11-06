import numpy as np
import pytest

from syne_tune.optimizer.schedulers.searchers.bore.de import (
    DifferentialevolutionOptimizer,
    STRATEGIES,
)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_de_stays_in_bounds_and_improves(strategy):
    np.random.seed(0)
    lower = np.zeros(3)
    upper = np.ones(3)

    def f(x):
        return float(np.sum((x - 0.3) ** 2))

    de = DifferentialevolutionOptimizer(f, lower, upper, fevals=200, strategy=strategy)
    best, traj = de.run()
    assert best.shape == (3,)
    assert np.all(best >= lower) and np.all(best <= upper)
    assert traj[-1] <= traj[0]
    assert f(best) == pytest.approx(traj[-1])


def test_de_unknown_strategy():
    with pytest.raises(ValueError):
        DifferentialevolutionOptimizer(lambda x: 0.0, np.zeros(2), np.ones(2), 100, "x")
