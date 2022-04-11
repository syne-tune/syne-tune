import logging
import numpy as np
import pandas as pd
import syne_tune.config_space as sp

from syne_tune.blackbox_repository import load, add_surrogate, \
    BlackboxRepositoryBackend, UserBlackboxBackend
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular

from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.baselines import ASHA
from syne_tune import StoppingCriterion, Tuner


def example_blackbox():
    n = 100
    x1 = np.arange(n)
    x2 = np.arange(n)[::-1]
    hyperparameters = pd.DataFrame(data=np.stack([x1, x2]).T, columns=["hp_x1", "hp_x2"])
    config_space = {
        "hp_x1": sp.randint(0, n),
        "hp_x2": sp.randint(0, n),
    }
    n_epochs = 100
    cs_fidelity = {
        'hp_epoch': sp.randint(0, n_epochs),
    }
    num_seeds = 1
    num_objectives = 3

    objectives_evaluations = np.random.rand(
        len(hyperparameters),
        num_seeds,
        n_epochs,
        num_objectives
    )
    # dummy runtime
    for t in range(0, n_epochs):
        objectives_evaluations[:, :, t, 1] = 60 * (t + 1)
    return add_surrogate(BlackboxTabular(
        hyperparameters=hyperparameters,
        configuration_space=config_space,
        fidelity_space=cs_fidelity,
        objectives_evaluations=objectives_evaluations,
        objectives_names=["metric_error", "runtime", "gpu_usage"]
    ))


def simulate_benchmark(blackbox, trial_backend, metric):
    # Random search without stopping
    scheduler = ASHA(
        blackbox.configuration_space,
        max_t=max(blackbox.fidelity_values),
        resource_attr=next(iter(blackbox.fidelity_space.keys())),
        mode='min',
        metric=metric,
        random_seed=31415927
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=7200)

    # It is important to set `sleep_time` to 0 here (mandatory for simulator backend)
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


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    n_workers = 4

    ## example of loading nasbench201 and then simulating tuning
    blackbox_name, dataset, metric = "nasbench201", "cifar100", "metric_valid_error"
    # Note: The nasbench201 blackbox does not have an `elapsed_time_attr`,
    # but only a `time_this_resource_attr`. The former is appended as the
    # cumulative sum of the latter in the backend implementation.
    time_this_resource_attr = 'metric_runtime'
    elapsed_time_attr = 'metric_elapsed_time'
    blackbox = load(blackbox_name)[dataset]
    trial_backend = BlackboxRepositoryBackend(
        blackbox_name=blackbox_name,
        dataset=dataset,
        elapsed_time_attr=elapsed_time_attr,
        time_this_resource_attr=time_this_resource_attr,
    )
    simulate_benchmark(blackbox=blackbox, trial_backend=trial_backend, metric=metric)

    ## example of loading a blackbox with custom code and then simulating tuning
    metric = "metric_error"
    blackbox = example_blackbox()
    trial_backend = UserBlackboxBackend(
        blackbox=blackbox,
        elapsed_time_attr="runtime",
    )
    simulate_benchmark(blackbox=blackbox, trial_backend=trial_backend, metric=metric)
