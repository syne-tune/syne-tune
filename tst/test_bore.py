# this test is commented as an import is causing the whole test suite to fail (without running)
# depending on the numpy/GPy versions installed.
"""
import numpy as np

from datetime import datetime

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.schedulers.searchers.bore.bore import Bore
from syne_tune.config_space import randint

max_steps = 10

config_space = {
    "steps": max_steps,
    "width": randint(0, 20),
}
time_attr = "step"
metric1 = "mean_loss"
metric2 = "cost"


def make_trial(trial_id: int):
    return Trial(
        trial_id=trial_id,
        config={"steps": 0, "width": trial_id},
        creation_time=datetime.now(),
    )


def test_bore_xgboost():
    searcher = Bore(config_space, metric='accuracy', acq_optimizer='de')

    for i in range(10):
        config = searcher.get_config(trial_id=i)
        result = {'accuracy': np.random.rand(), 'time': 1.0, 'step': 2}

        searcher.on_trial_result('%d' % i, config, result, update=True)

    config = searcher.get_config(trial_id=10)


def test_bore_gp():
    searcher = Bore(config_space, metric='accuracy', classifier='gp')

    for i in range(10):
        config = searcher.get_config(trial_id=i)
        result = {'accuracy': np.random.rand(), 'time': 1.0, 'step': 2}

        searcher.on_trial_result('%d' % i, config, result, update=True)

    config = searcher.get_config(trial_id=10)


def test_bore_mlp():
    searcher = Bore(config_space, metric='accuracy', classifier='mlp')

    for i in range(10):
        config = searcher.get_config(trial_id=i)
        result = {'accuracy': np.random.rand(), 'time': 1.0, 'step': 2}

        searcher.on_trial_result('%d' % i, config, result, update=True)

    config = searcher.get_config(trial_id=10)


def test_bore_rf():
    searcher = Bore(config_space, metric='accuracy', classifier='rf')

    for i in range(10):
        config = searcher.get_config(trial_id=i)
        result = {'accuracy': np.random.rand(), 'time': 1.0, 'step': 2}

        searcher.on_trial_result('%d' % i, config, result, update=True)

    config = searcher.get_config(trial_id=10)
"""
