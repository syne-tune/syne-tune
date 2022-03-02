# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import argparse

from syne_tune.config_space import add_to_argparse, randint, uniform, \
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
