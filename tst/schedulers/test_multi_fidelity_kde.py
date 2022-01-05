import numpy as np

from syne_tune.optimizer.schedulers.searchers.multi_fidelity_kde_searcher import MultiFidelityKernelDensityEstimator

import syne_tune.search_space as sp


config_space = {
    "steps": 100,
    "x": sp.randint(0, 20),
    "y": sp.uniform(0, 1),
    "z": sp.choice(["a", "b", "c"]),
}
metric = "objective"
resource_attr = 'epochs'
searcher = MultiFidelityKernelDensityEstimator(configspace=config_space, mode="max", metric=metric,
                                               resource_attr=resource_attr)


def test_multi_fidelity_kde():
    num_trials = 50
    for i in range(num_trials):
        config = searcher.get_config()
        assert all(x in config.keys() for x in config_space.keys()), \
            "suggestion configuration should contains all keys of configspace."

    curr_rung_level = 0
    for r in [1, 3, 9]:
        for i in range(num_trials):
            for ri in range(curr_rung_level, r):
                searcher.on_trial_result(trial_id=i, config=config,
                                         result={metric: np.random.rand(), resource_attr: ri}, update=True)
            curr_rung_level = r