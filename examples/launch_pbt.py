import logging
from pathlib import Path

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.schedulers import PopulationBasedTraining
from syne_tune import Tuner
from syne_tune.config_space import loguniform
from syne_tune import StoppingCriterion


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    max_trials = 100

    config_space = {
        "lr": loguniform(0.0001, 0.02),
    }

    entry_point = (
        Path(__file__).parent / "training_scripts" / "pbt_example" / "pbt_example.py"
    )
    trial_backend = LocalBackend(entry_point=str(entry_point))

    mode = "max"
    metric = "mean_accuracy"
    time_attr = "training_iteration"
    population_size = 2

    pbt = PopulationBasedTraining(
        config_space=config_space,
        metric=metric,
        resource_attr=time_attr,
        population_size=population_size,
        mode=mode,
        max_t=200,
        perturbation_interval=1,
    )

    local_tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=pbt,
        stop_criterion=StoppingCriterion(max_wallclock_time=20),
        n_workers=population_size,
        results_update_interval=1,
    )

    local_tuner.run()
