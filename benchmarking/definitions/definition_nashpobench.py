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
from syne_tune.search_space import choice, uniform, loguniform, lograndint, \
    logfinrange, finrange

from benchmarking.blackbox_repository.conversion_scripts.scripts.fcnet_import \
    import METRIC_ELAPSED_TIME, METRIC_VALID_LOSS, RESOURCE_ATTR, \
    BLACKBOX_NAME


__config_space = {
    "hp_activation_fn_1": choice(["tanh", "relu"]),
    "hp_activation_fn_2": choice(["tanh", "relu"]),
    'hp_lr_schedule': choice(["cosine", "const"]),
}


_config_space = {
    True: dict(
        __config_space,
        hp_batch_size=lograndint(8, 64),
        hp_dropout_1=uniform(0.0, 0.6),
        hp_dropout_2=uniform(0.0, 0.6),
        hp_init_lr=loguniform(0.0005, 0.1),
        hp_n_units_1=lograndint(16, 512),
        hp_n_units_2=lograndint(16, 512)),
    False: dict(
        __config_space,
        hp_batch_size=logfinrange(8, 64, 4, cast_int=True),
        hp_dropout_1=finrange(0.0, 0.6, 3),
        hp_dropout_2=finrange(0.0, 0.6, 3),
        hp_init_lr=choice([0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]),
        hp_n_units_1=logfinrange(16, 512, 6, cast_int=True),
        hp_n_units_2=logfinrange(16, 512, 6, cast_int=True))
}


def nashpobench_default_params(params=None):
    return {
        'max_resource_level': 100,
        'grace_period': 1,
        'reduction_factor': 3,
        'instance_type': 'ml.m5.large',
        'num_workers': 4,
        'framework': 'PyTorch',
        'framework_version': '1.6',
        'dataset_name': 'protein_structure',
        'interpolate_blackbox': True,
    }


def nashpobench_benchmark(params):
    """
    The underlying tabulated blackbox does not have an `elapsed_time_attr`,
    but only a `time_this_resource_attr`.

    The boolean parameter `interpolate_blackbox` decides whether the
    tabulated blackbox values are interpolated (using a random forest), in
    which case the hyperparameter ranges are intervals, or whether the
    ranges are only exactly covering the tabulated grid.
    Note that the latter leads to a larger encoded dimension (which can
    be a problem for Bayesian optimization), because one of the numerical
    parameters has to be encoded as categorical.

    """
    interpolate_blackbox = params['interpolate_blackbox']
    config_space = dict(
        _config_space[interpolate_blackbox],
        epochs=params['max_resource_level'],
        dataset_name=params['dataset_name'],
        blackbox_repo_s3_root=params.get('blackbox_repo_s3_root'))
    surrogate = 'random_forest' if interpolate_blackbox else None
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
        'surrogate': surrogate,
        'time_this_resource_attr': METRIC_ELAPSED_TIME,
    }
