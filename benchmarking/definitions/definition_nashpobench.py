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
from syne_tune.blackbox_repository.conversion_scripts.scripts.fcnet_import \
    import METRIC_ELAPSED_TIME, METRIC_VALID_LOSS, RESOURCE_ATTR, \
    BLACKBOX_NAME, MAX_RESOURCE_LEVEL, CONFIGURATION_SPACE, NUM_UNITS_1, \
    NUM_UNITS_2

from syne_tune.config_space import choice, uniform, loguniform, lograndint


DATASET_NAMES = [
    "protein_structure",
    "naval_propulsion",
    "parkinsons_telemonitoring",
    "slice_localization",
]


def _config_space(interpolate_blackbox: bool) -> dict:
    """
    :param interpolate_blackbox: Are blackbox values interpolated by a
        surrogate model?
    :return: Configuration space to be used
    """
    if interpolate_blackbox:
        return dict(
            hp_activation_fn_1=choice(["tanh", "relu"]),
            hp_activation_fn_2=choice(["tanh", "relu"]),
            hp_lr_schedule=choice(["cosine", "const"]),
            hp_batch_size=lograndint(8, 64),
            hp_dropout_1=uniform(0.0, 0.6),
            hp_dropout_2=uniform(0.0, 0.6),
            hp_init_lr=loguniform(0.0005, 0.1),
            hp_n_units_1=lograndint(16, 512),
            hp_n_units_2=lograndint(16, 512),
        )
    else:
        return CONFIGURATION_SPACE


def nashpobench_default_params(params=None):
    return {
        'max_resource_level': MAX_RESOURCE_LEVEL,
        'grace_period': 1,
        'reduction_factor': 3,
        'instance_type': 'ml.m5.large',
        'num_workers': 4,
        'framework': 'PyTorch',
        'framework_version': '1.6',
        'dataset_name': 'protein_structure',
        'interpolate_blackbox': False,
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
        _config_space(interpolate_blackbox),
        epochs=params['max_resource_level'],
        dataset_name=params['dataset_name'])
    if interpolate_blackbox:
        surrogate = 'RandomForestRegressor'
        surrogate_kwargs = dict(max_samples=0.005, bootstrap=True)
    else:
        surrogate = None
        surrogate_kwargs = None
    return {
        'script': None,
        'metric': METRIC_VALID_LOSS,
        'mode': 'min',
        'resource_attr': RESOURCE_ATTR,
        'elapsed_time_attr': METRIC_ELAPSED_TIME,
        'max_resource_attr': 'epochs',
        'config_space': config_space,
        'cost_model': get_cost_model(params),
        'supports_simulated': True,
        'blackbox_name': BLACKBOX_NAME,
        'surrogate': surrogate,
        'surrogate_kwargs': surrogate_kwargs,
    }


# See Table 1 in https://arxiv.org/abs/1905.04970
_NUM_FEATURES = {
    'protein_structure': 9,
    'naval_propulsion': 15,
    'parkinsons_telemonitoring': 20,
    'slice_localization': 385,
}


def get_cost_model(params):
    """
    This cost model ignores the batch size, but depends on the number of units
    in the two layers only.
    """
    try:
        from syne_tune.optimizer.schedulers.searchers.bayesopt.models.cost.linear_cost_model \
            import FixedLayersMLPCostModel

        num_inputs = _NUM_FEATURES[params['dataset_name']]
        num_outputs = 1  # All benchmarks are regression problems
        num_units_keys = [NUM_UNITS_1, NUM_UNITS_2]
        expected_hidden_layer_width, exp_vals = \
            FixedLayersMLPCostModel.get_expected_hidden_layer_width(
                CONFIGURATION_SPACE, num_units_keys)
        return FixedLayersMLPCostModel(
            num_inputs=num_inputs, num_outputs=num_outputs,
            num_units_keys=num_units_keys,
            expected_hidden_layer_width=expected_hidden_layer_width)
    except Exception:
        return None
