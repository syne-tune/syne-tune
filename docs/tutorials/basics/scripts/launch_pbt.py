import logging
from pathlib import Path

from syne_tune.search_space import randint, uniform, loguniform
from syne_tune.backend.local_backend import LocalBackend
from syne_tune.optimizer.schedulers.pbt import PopulationBasedTraining
from syne_tune.tuner import Tuner
from syne_tune.stopping_criterion import StoppingCriterion


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    random_seed = 31415927
    n_workers = 4
    max_wallclock_time = 3 * 3600  # Run for 3 hours
    max_resource_level = 81  # Maximum number of training epochs

    # Here, we specify the training script we want to tune
    # - `mode` and `metric` must match what is reported in the training script
    # - Metrics need to be reported after each epoch, `resource_attr` must match
    #   what is reported in the training script
    entry_point = str(Path(__file__).parent / "traincode_report_withcheckpointing.py")
    mode = 'max'
    metric = 'accuracy'
    resource_attr = 'epoch'
    max_resource_attr = 'epochs'

    # Search space (or configuration space)
    # For each tunable parameter, need to define type, range, and encoding
    # (linear, logarithmic)
    config_space = {
        'n_units_1': randint(4, 1024),
        'n_units_2': randint(4, 1024),
        'batch_size': randint(8, 128),
        'dropout_1': uniform(0, 0.99),
        'dropout_2': uniform(0, 0.99),
        'learning_rate': loguniform(1e-6, 1),
        'weight_decay': loguniform(1e-8, 1),
    }

    # Additional fixed parameters
    config_space.update({
        max_resource_attr: max_resource_level,
        'dataset_path': './',
    })

    # Local back-end: Responsible for scheduling trials
    # The local back-end runs trials as sub-processes on a single instance
    backend = LocalBackend(entry_point=entry_point)

    # Scheduler: Population-based training (PBT)
    # PBT aims to tune hyperparameters while training. A population of
    # `n_workers` models are trained in parallel. Each `perturbation_interval`
    # epochs, configs may be perturbed or resampled, and model weights may be
    # transferred.
    scheduler = PopulationBasedTraining(
        config_space,
        resource_attr=resource_attr,
        max_resource_attr=max_resource_attr,
        population_size=n_workers,
        perturbation_interval=3,
        mode=mode,
        metric=metric,
        random_seed=random_seed,
    )

    # The experiment is stopped after `max_wallclock_time` seconds
    stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)

    # Everything comes together in the tuner
    tuner = Tuner(
        backend=backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
    )

    tuner.run()
