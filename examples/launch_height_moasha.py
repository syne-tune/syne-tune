"""
Example showing how to tune multiple objectives at once of an artificial function.
"""
import logging
from pathlib import Path

import numpy as np

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.schedulers.multiobjective.moasha import MOASHA
from syne_tune import Tuner, StoppingCriterion
from syne_tune.config_space import uniform


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(0)

    max_steps = 27
    n_workers = 4

    config_space = {
        "steps": max_steps,
        "theta": uniform(0, np.pi / 2),
        "sleep_time": 0.01,
    }
    entry_point = (
        Path(__file__).parent
        / "training_scripts"
        / "mo_artificial"
        / "mo_artificial.py"
    )
    mode = "min"

    np.random.seed(0)
    scheduler = MOASHA(
        max_t=max_steps,
        resource_attr="step",
        metrics=["y1", "y2"],
        config_space=config_space,
    )
    trial_backend = LocalBackend(entry_point=str(entry_point))

    stop_criterion = StoppingCriterion(max_wallclock_time=20)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        sleep_time=0.5,
    )
    tuner.run()
