"""
This example shows how to run RandomSearch on Benchmarks from HPO-B by Pineda et al.
We use a XGBoost surrogate model to predict the performance of unobserved hyperparameter configurations.
"""
import logging
from syne_tune.blackbox_repository import (
    load_blackbox,
    BlackboxRepositoryBackend,
)

from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.baselines import RandomSearch, CQR
from syne_tune import StoppingCriterion, Tuner


def simulate_benchmark(blackbox, trial_backend, metric):
    max_resource_attr = "epochs"
    scheduler = CQR(
        config_space=blackbox.configuration_space_with_max_resource_attr(
            max_resource_attr
        ),
        metric=metric,
        random_seed=31415927,
        do_minimize=False
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=600)

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

    metric = "metric_accuracy"
    blackbox = load_blackbox("hpob_4796", local_files_only=True)["3549"]
    trial_backend = BlackboxRepositoryBackend(
        blackbox_name="hpob_4796",
        dataset="3549",
        elapsed_time_attr="metric_elapsed_time",
        surrogate="XGBRegressor",
    )
    simulate_benchmark(blackbox=blackbox, trial_backend=trial_backend, metric=metric)
