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

from sklearn.ensemble import RandomForestRegressor

from syne_tune.search_space import choice, randint, uniform, loguniform

from benchmarking.blackbox_repository.conversion_scripts.scripts.fcnet_import import \
    METRIC_ELAPSED_TIME, METRIC_VALID_LOSS, RESOURCE_ATTR, BLACKBOX_NAME


_config_space = {
    "hp_activation_fn_1": choice(["tanh", "relu"]),
    "hp_activation_fn_2": choice(["tanh", "relu"]),
    "hp_batch_size": randint(8, 64),
    "hp_dropout_1": uniform(0.0, 0.6),
    "hp_dropout_2": uniform(0.0, 0.6),
    "hp_init_lr": loguniform(0.0005, 0.1),
    'hp_lr_schedule': choice(["cosine", "const"]),
    'hp_n_units_1': randint(16, 512),
    'hp_n_units_2': randint(16, 512),
}


def nashpobench_default_params(params=None):
    dont_sleep = str(
        params is not None and params.get('backend') == 'simulated')
    return {
        'max_resource_level': 100,
        'grace_period': 1,
        'reduction_factor': 3,
        'instance_type': 'ml.m5.large',
        'num_workers': 4,
        'framework': 'PyTorch',
        'framework_version': '1.6',
        'dataset_name': 'protein_structure',
        'dont_sleep': dont_sleep,
    }


def nashpobench_benchmark(params):
    """
    The underlying tabulated blackbox does not have an `elapsed_time_attr`,
    but only a `time_this_resource_attr`.

    """
    config_space = dict(
        _config_space,
        epochs=params['max_resource_level'],
        dataset_name=params['dataset_name'],
        dont_sleep=params['dont_sleep'],
        blackbox_repo_s3_root=params.get('blackbox_repo_s3_root'))
    return {
        'script': None,
        'metric': METRIC_VALID_LOSS,
        'mode': 'min',
        'resource_attr': RESOURCE_ATTR,
        'elapsed_time_attr': METRIC_ELAPSED_TIME,
        'max_resource_attr': 'epochs',
        'config_space': config_space,
        'cost_model': None,
        'supports_simulated': True,
        'blackbox_name': BLACKBOX_NAME,
        'surrogate': RandomForestRegressor(max_samples=0.005, bootstrap=True),
        'time_this_resource_attr': METRIC_ELAPSED_TIME,
    }

