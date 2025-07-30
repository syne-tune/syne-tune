"""
This example shows how to run random search on the TabRepo benchmark. For more details see: 

TabRepo: A Large Scale Repository of Tabular Model Evaluations and its Auto{ML} Applications},
David Salinas and Nick Erickson
AutoML Conference 2024 (ABCD Track)
We use a XGBoost surrogate model to predict the performance of unobserved hyperparameter configurations.
"""
import logging
from syne_tune.blackbox_repository import (
    load_blackbox,
    BlackboxRepositoryBackend,
)

from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune import StoppingCriterion, Tuner


def simulate_benchmark(blackbox, trial_backend, metric):
    max_resource_attr = "epochs"
    scheduler = RandomSearch(
        config_space=blackbox.configuration_space_with_max_resource_attr(
            max_resource_attr
        ),
        metrics=[metric],
        random_seed=31415927,
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=7200)

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        sleep_time=0,
        callbacks=[SimulatorCallback()],
    )
    tuner.run()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    n_workers = 4

    metric = "metric_error"
    blackbox = load_blackbox("tabrepo_ExtraTrees")["arcene"]
    trial_backend = BlackboxRepositoryBackend(
        blackbox_name="tabrepo_ExtraTrees",
        dataset="arcene",
        elapsed_time_attr="metric_elapsed_time",
        surrogate="XGBRegressor",
    )
    simulate_benchmark(blackbox=blackbox, trial_backend=trial_backend, metric=metric)
