import logging
from syne_tune.blackbox_repository import (
    load_blackbox,
    BlackboxRepositoryBackend,
    UserBlackboxBackend, add_surrogate,
)

from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune import StoppingCriterion, Tuner


def simulate_benchmark(blackbox, trial_backend, metric):
    # Asynchronous successive halving
    max_resource_attr = "epochs"
    scheduler = RandomSearch(
        config_space=blackbox.configuration_space_with_max_resource_attr(
            max_resource_attr
        ),
        max_resource_attr=max_resource_attr,
        metric=metric,
        random_seed=31415927,
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=7200)

    # It is important to set ``sleep_time`` to 0 here (mandatory for simulator backend)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        sleep_time=0,
        # This callback is required in order to make things work with the
        # simulator callback. It makes sure that results are stored with
        # simulated time (rather than real time), and that the time_keeper
        # is advanced properly whenever the tuner loop sleeps
        callbacks=[SimulatorCallback()],
    )
    tuner.run()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    n_workers = 4

    metric = "metric_accuracy"
    blackbox = load_blackbox('hpob_4796')['3549']
    trial_backend = BlackboxRepositoryBackend(
        blackbox_name='hpob_4796',
        dataset='3549',
        elapsed_time_attr='metric_elapsed_time'
    )
    simulate_benchmark(blackbox=blackbox, trial_backend=trial_backend, metric=metric)


