"""
Example for running CQR on NASBench201.
"""
import logging

from syne_tune.experiments.benchmark_definitions.nas201 import nas201_benchmark
from syne_tune.blackbox_repository import BlackboxRepositoryBackend
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.legacy_baselines import ASHACQR
from syne_tune import Tuner, StoppingCriterion
from syne_tune.experiments import load_experiment


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)

    n_workers = 4
    dataset_name = "cifar100"
    benchmark = nas201_benchmark(dataset_name)

    # simulator backend specialized to tabulated blackboxes
    max_resource_attr = benchmark.max_resource_attr
    trial_backend = BlackboxRepositoryBackend(
        blackbox_name=benchmark.blackbox_name,
        elapsed_time_attr=benchmark.elapsed_time_attr,
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
        mode=benchmark.mode,
        metric=benchmark.metric,
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
