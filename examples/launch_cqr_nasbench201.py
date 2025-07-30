"""
Example for running CQR on NASBench201.
"""
import logging

from syne_tune.blackbox_repository import BlackboxRepositoryBackend
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.legacy_baselines import ASHACQR
from syne_tune import Tuner, StoppingCriterion
from syne_tune.experiments import load_experiment


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)

    n_workers = 4
    dataset_name = "cifar100"
    max_resource_attr = "hp_epoch"
    # simulator backend specialized to tabulated blackboxes
    trial_backend = BlackboxRepositoryBackend(
        blackbox_name="nasbench201",
        elapsed_time_attr="metric_elapsed_time",
        max_resource_attr=max_resource_attr,
        dataset=dataset_name,
    )

    blackbox = trial_backend.blackbox
    scheduler = ASHACQR(
        config_space=blackbox.configuration_space_with_max_resource_attr(
            max_resource_attr
        ),
        max_resource_attr=max_resource_attr,
        resource_attr=blackbox.fidelity_name(),
        mode="min",
        metric="metric_valid_error",
    )

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_wallclock_time=3600 * 3),
        n_workers=n_workers,
        sleep_time=0,
        # this callback is required in order to make things work with the
        # simulator callback. It makes sure that results are stored with
        # simulated time (rather than real time), and that the time_keeper
        # is advanced properly whenever the tuner loop sleeps
        callbacks=[SimulatorCallback()],
    )

    tuner.run()

    tuning_experiment = load_experiment(tuner.name)
    tuning_experiment.plot()
