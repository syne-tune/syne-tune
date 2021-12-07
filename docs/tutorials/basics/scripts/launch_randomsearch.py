import logging
from pathlib import Path

from syne_tune.search_space import randint, uniform, loguniform
from syne_tune.backend.local_backend import LocalBackend
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
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
    entry_point = str(Path(__file__).parent / "traincode_report_end.py")
    mode = 'max'
    metric = 'accuracy'
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
    # [1]
    config_space.update({
        max_resource_attr: max_resource_level,
        'dataset_path': './',
    })

    # Local back-end: Responsible for scheduling trials
    # The local back-end runs trials as sub-processes on a single instance
    # [2]
    backend = LocalBackend(entry_point=entry_point)

    # Scheduler:
    # The `FIFOScheduler` starts a trial whenever a worker is free. It does
    # not stop or pause trials, they always run to the end.
    # We configure this scheduler with random search: configurations for new
    # trials are drawn at random
    # [3]
    searcher = 'random'
    scheduler = FIFOScheduler(
        config_space,
        searcher=searcher,
        mode=mode,
        metric=metric,
        random_seed=random_seed,
    )

    # The experiment is stopped after `max_wallclock_time` seconds
    # [4]
    stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)

    # Everything comes together in the tuner
    tuner = Tuner(
        backend=backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
    )

    tuner.run()
