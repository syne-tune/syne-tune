"""
This launches a local HPO tuning the discount factor of PPO on cartpole.
To run this example, you should have installed dependencies in `requirements.txt`.
"""
import logging
from pathlib import Path

import numpy as np

from syne_tune.backend import LocalBackend
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.baselines import ASHA
import syne_tune.config_space as sp
from syne_tune import Tuner

if __name__ == '__main__':

    logging.getLogger().setLevel(logging.DEBUG)
    np.random.seed(0)
    max_steps = 100
    trial_backend = LocalBackend(entry_point=Path(__file__).parent / "training_scripts" / "rl_cartpole" / "train_cartpole.py")

    scheduler = ASHA(
        config_space={
            "gamma": sp.uniform(0.5, 0.99),
            "lr": sp.loguniform(1e-6, 1e-3),
        },
        metric="episode_reward_mean",
        mode="max",
        max_t=100,
        resource_attr="training_iter",
        search_options={'debug_log': False},
    )

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        # tune for 3 minutes
        stop_criterion=lambda status: status.wallclock_time > 60,
        n_workers=2,
    )

    tuner.run()

    tuning_experiment = load_experiment(tuner.name)

    print(f"best result found: {tuning_experiment.best_config()}")

    tuning_experiment.plot()