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
"""
Two-layer MLP trained on Fashion MNIST
"""
from pathlib import Path

from benchmarking.training_scripts.mlp_on_fashion_mnist.mlp_on_fashion_mnist import \
    NUM_UNITS_1, NUM_UNITS_2, _config_space, METRIC_NAME, RESOURCE_ATTR, \
    ELAPSED_TIME_ATTR


def mlp_fashionmnist_default_params(params=None):
    return {
        'max_resource_level': 81,
        'grace_period': 1,
        'reduction_factor': 3,
        'instance_type': 'ml.c5.4xlarge',
        'num_workers': 4,
        'framework': 'PyTorch',
        'framework_version': '1.6',
        'dataset_path': './',
        'report_current_best': 'False',
    }


def mlp_fashionmnist_benchmark(params):
    config_space = dict(
        _config_space,
        dataset_path=params['dataset_path'],
        epochs=params['max_resource_level'],
        report_current_best=params['report_current_best'])
    return {
        'script': Path(__file__).parent.parent / "training_scripts" /
                  "mlp_on_fashion_mnist" / "mlp_on_fashion_mnist.py",
        'metric': METRIC_NAME,
        'mode': 'max',
        'resource_attr': RESOURCE_ATTR,
        'elapsed_time_attr': ELAPSED_TIME_ATTR,
        'max_resource_attr': 'epochs',
        'map_reward': '1_minus_x',
        'config_space': config_space,
        'cost_model': get_cost_model(params),
    }


def get_cost_model(params):
    """
    This cost model ignores the batch size, but depends on the number of units
    in the two layers only.
    """
    try:
        from syne_tune.optimizer.schedulers.searchers.bayesopt.models.cost.linear_cost_model \
            import FixedLayersMLPCostModel

        num_inputs = 28 * 28
        num_outputs = 10
        num_units_keys = [NUM_UNITS_1, NUM_UNITS_2]
        expected_hidden_layer_width, exp_vals = \
            FixedLayersMLPCostModel.get_expected_hidden_layer_width(
                _config_space, num_units_keys)
        return FixedLayersMLPCostModel(
            num_inputs=num_inputs, num_outputs=num_outputs,
            num_units_keys=num_units_keys,
            expected_hidden_layer_width=expected_hidden_layer_width)
    except Exception:
        return None
