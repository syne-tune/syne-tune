import numpy as np

from syne_tune.optimizer.schedulers.searchers.kde_searcher import KernelDensityEstimator

import syne_tune.search_space as sp


config_space = {
    "steps": 100,
    "x": sp.randint(0, 20),
    "y": sp.uniform(0, 1),
    "z": sp.choice(["a", "b", "c"]),
}
metric = "objective"

searcher = KernelDensityEstimator(configspace=config_space, mode="max", metric=metric)


def test_kde():
    for i in range(50):
        config = searcher.get_config()
        assert all(x in config.keys() for x in config_space.keys()), \
            "suggestion configuration should contains all keys of configspace."

        searcher.on_trial_result(trial_id=i, config=config, result={metric: np.random.rand()}, update=True)
