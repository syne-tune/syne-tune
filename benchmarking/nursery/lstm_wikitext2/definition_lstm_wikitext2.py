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
Example that reproduces the LSTM on WikiText2 benchmark from AutoGluonExperiments repo
"""
from pathlib import Path

from benchmarking.utils import get_cost_model_for_batch_size
from benchmarking.nursery.lstm_wikitext2.lstm_wikitext2 import \
    BATCH_SIZE_LOWER, BATCH_SIZE_UPPER, BATCH_SIZE_KEY, _config_space, \
    METRIC_NAME, RESOURCE_ATTR, ELAPSED_TIME_ATTR


def lstm_wikitext2_default_params(params=None):
    if params is not None and params.get('backend') == 'sagemaker':
        instance_type = 'ml.g4dn.xlarge'
        num_workers = 8
    else:
        # For local backend, GPU cores serve different workers, so we
        # need more memory
        instance_type = 'ml.g4dn.12xlarge'
        num_workers = 4
    return {
        'max_resource_level': 81,
        'grace_period': 1,
        'reduction_factor': 3,
        'instance_type': instance_type,
        'num_workers': num_workers,
        'framework': 'PyTorch',
        'framework_version': '1.6',
        'report_current_best': 'False',
        'dataset_path': './',
        'cost_model_type': 'quadratic_spline',
    }


def lstm_wikitext2_benchmark(params):
    config_space = dict(
        _config_space,
        dataset_path=params['dataset_path'],
        epochs=params['max_resource_level'],
        report_current_best=params['report_current_best'])
    return {
        'script': "lstm_wikitext2.py",
        'metric': METRIC_NAME,
        'mode': 'max',
        'resource_attr': RESOURCE_ATTR,
        'elapsed_time_attr': ELAPSED_TIME_ATTR,
        'max_resource_attr': 'epochs',
        'map_reward': 'minus_x',
        'config_space': config_space,
        'cost_model': get_cost_model_for_batch_size(
            params, batch_size_key=BATCH_SIZE_KEY,
            batch_size_range=(BATCH_SIZE_LOWER, BATCH_SIZE_UPPER)),
    }
