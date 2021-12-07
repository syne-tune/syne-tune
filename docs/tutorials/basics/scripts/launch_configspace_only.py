import logging
from pathlib import Path

from syne_tune.search_space import randint, uniform, loguniform


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # [1]
    random_seed = 31415927
    n_workers = 4
    max_wallclock_time = 3 * 3600  # Run for 3 hours
    max_resource_level = 81  # Maximum number of training epochs

    # Here, we specify the training script we want to tune
    # - `mode` and `metric` must match what is reported in the training script
    # [2]
    entry_point = str(Path(__file__).parent / "traincode_report_end.py")
    mode = 'max'
    metric = 'accuracy'
    max_resource_attr = 'epochs'

    # Search space (or configuration space)
    # For each tunable parameter, need to define type, range, and encoding
    # (linear, logarithmic)
    # [3]
    config_space = {
        'n_units_1': randint(4, 1024),
        'n_units_2': randint(4, 1024),
        'batch_size': randint(8, 128),
        'dropout_1': uniform(0, 0.99),
        'dropout_2': uniform(0, 0.99),
        'learning_rate': loguniform(1e-6, 1),
        'weight_decay': loguniform(1e-8, 1),
    }
