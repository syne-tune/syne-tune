"""
Example showing how to tune given a script ("training_script.py") that takes input hyperparameters
as a file rather than command line arguments.
Note that this approach only works with `LocalBackend` at the moment.
"""
from pathlib import Path

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint
from syne_tune.optimizer.baselines import RandomSearch

if __name__ == '__main__':
    config_space = {"x": randint(0, 10)}
    tuner = Tuner(
        scheduler=RandomSearch(config_space=config_space, metric="error"),
        trial_backend=LocalBackend(entry_point=str(Path(__file__).parent / "training_script.py")),
        stop_criterion=StoppingCriterion(max_wallclock_time=20),
        n_workers=2,
    )
    tuner.run()