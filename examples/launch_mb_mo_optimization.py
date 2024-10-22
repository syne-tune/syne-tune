from pathlib import Path

import numpy as np

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint, uniform
from syne_tune.optimizer.baselines import MORandomScalarizationBayesOpt


def main():
    random_seed = 6287623
    # Hyperparameter configuration space
    config_space = {
        "steps": randint(0, 100),
        "theta": uniform(0, np.pi / 2),
        "sleep_time": 0.01,
    }
    metrics = ["y1", "y2"]
    modes = ["min", "min"]

    # Creates a FIFO scheduler with a ``MultiObjectiveMultiSurrogateSearcher``. The
    # latter is configured by one default GP surrogate per objective, and with the
    # ``MultiObjectiveLCBRandomLinearScalarization`` acquisition function.
    scheduler = MORandomScalarizationBayesOpt(
        config_space=config_space,
        metric=metrics,
        mode=modes,
        random_seed=random_seed,
    )

    entry_point = str(
        Path(__file__).parent
        / "training_scripts"
        / "mo_artificial"
        / "mo_artificial.py"
    )
    tuner = Tuner(
        trial_backend=LocalBackend(entry_point=entry_point),
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_wallclock_time=30),
        n_workers=1,  # how many trials are evaluated in parallel
    )
    tuner.run()


if __name__ == "__main__":
    main()
