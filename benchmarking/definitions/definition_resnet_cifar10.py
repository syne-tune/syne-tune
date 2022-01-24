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
from pathlib import Path

from benchmarking.utils import get_cost_model_for_batch_size
from benchmarking.training_scripts.resnet_cifar10.resnet_cifar10 import \
    BATCH_SIZE_LOWER, BATCH_SIZE_UPPER, BATCH_SIZE_KEY, METRIC_NAME, \
    RESOURCE_ATTR, ELAPSED_TIME_ATTR, _config_space


def resnet_cifar10_default_params(params=None):
    if params is not None and params.get('backend') == 'sagemaker':
        instance_type = 'ml.g4dn.xlarge'
    else:
        # For local backend, GPU cores serve different workers, so we
        # need more memory
        instance_type = 'ml.g4dn.12xlarge'
    return {
        'epochs': 27,
        'grace_period': 1,
        'reduction_factor': 3,
        'instance_type': instance_type,
        'num_workers': 4,
        'framework': 'PyTorch',
        'framework_version': '1.6',
        'num_gpus': 1,
        'cost_model_type': 'quadratic_spline',
    }


def resnet_cifar10_benchmark(params):
    config_space = dict(
        _config_space,
        epochs=params['max_resource_level'],
        dataset_path=params['dataset_path'],
        num_gpus=params['num_gpus'])
    return {
        'script': Path(__file__).parent.parent.parent / "examples" /
                  "training_scripts" / "resnet_cifar10" / "resnet_cifar10.py",
        'metric': METRIC_NAME,
        'mode': 'max',
        'resource_attr': RESOURCE_ATTR,
        'elapsed_time_attr': ELAPSED_TIME_ATTR,
        'max_resource_attr': 'epochs',
        'map_reward': '1_minus_x',
        'config_space': config_space,
        'cost_model': get_cost_model_for_batch_size(
            params, batch_size_key=BATCH_SIZE_KEY,
            batch_size_range=(BATCH_SIZE_LOWER, BATCH_SIZE_UPPER)),
    }
