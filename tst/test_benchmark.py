import argparse

from sagemaker_tune.search_space import add_to_argparse, randint, uniform, \
    loguniform


def test_add_to_argparse():
    _config_space = {
        'n_units_1': randint(4, 1024),
        'n_units_2': randint(4, 1024),
        'batch_size': randint(8, 128),
        'dropout_1': uniform(0, 0.99),
        'dropout_2': uniform(0, 0.99),
        'learning_rate': loguniform(1e-6, 1),
        'wd': loguniform(1e-8, 1)}

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_log', action='store_true')
    add_to_argparse(parser, _config_space)

    _config = {
        'n_units_1': 6,
        'n_units_2': 100,
        'batch_size': 32,
        'dropout_1': 0.5,
        'dropout_2': 0.9,
        'learning_rate': 0.001,
        'wd': 0.25}

    args, _ = parser.parse_known_args(
        [f"--{k}={v}" for k, v in _config.items()])
    config=vars(args)
    for k, v in _config.items():
        assert k in config, f"{k} not in config"
        assert config[k] == v, \
            f"{config[k]} = config[{k}] != _config[{k}] = {v}"
    assert 'debug_log' in config
    assert not config['debug_log']
