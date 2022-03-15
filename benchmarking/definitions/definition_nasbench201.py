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
from syne_tune.config_space import choice
from benchmarking.blackbox_repository.conversion_scripts.scripts.nasbench201_import import \
    CONFIG_KEYS, METRIC_VALID_ERROR, METRIC_TIME_THIS_RESOURCE, \
    RESOURCE_ATTR, BLACKBOX_NAME


DATASET_NAMES = [
    "cifar10",
    "cifar100",
    "ImageNet16-120",
]


METRIC_ELAPSED_TIME = 'metric_elapsed_time'


# First is default value
x_range = ['skip_connect',
           'none',
           'nor_conv_1x1',
           'nor_conv_3x3',
           'avg_pool_3x3']


_config_space = {k: choice(x_range) for k in CONFIG_KEYS}


def nasbench201_default_params(params=None):
    dont_sleep = str(
        params is not None and params.get('backend') == 'simulated')
    return {
        'max_resource_level': 200,
        'grace_period': 1,
        'reduction_factor': 3,
        'instance_type': 'ml.m5.large',
        'num_workers': 4,
        'framework': 'PyTorch',
        'framework_version': '1.6',
        'dataset_name': 'cifar10',
        'dont_sleep': dont_sleep,
        'cost_model_type': 'linear',
    }


def nasbench201_benchmark(params):
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
        'metric': METRIC_VALID_ERROR,
        'mode': 'min',
        'resource_attr': RESOURCE_ATTR,
        'elapsed_time_attr': METRIC_ELAPSED_TIME,
        'max_resource_attr': 'epochs',
        'config_space': config_space,
        'cost_model': _get_cost_model(params),
        'supports_simulated': True,
        'blackbox_name': BLACKBOX_NAME,
        'time_this_resource_attr': METRIC_TIME_THIS_RESOURCE,
    }


def _get_cost_model(params):
    try:
        cost_model_type = params.get('cost_model_type')
        if cost_model_type is None:
            cost_model_type = 'linear'
        if cost_model_type.startswith('linear'):
            from syne_tune.optimizer.schedulers.searchers.bayesopt.models.cost.linear_cost_model \
                import NASBench201LinearCostModel

            map_config_values = {
                'skip_connect': NASBench201LinearCostModel.Op.SKIP_CONNECT,
                'none': NASBench201LinearCostModel.Op.NONE,
                'nor_conv_1x1': NASBench201LinearCostModel.Op.NOR_CONV_1x1,
                'nor_conv_3x3': NASBench201LinearCostModel.Op.NOR_CONV_3x3,
                'avg_pool_3x3': NASBench201LinearCostModel.Op.AVG_POOL_3x3,
            }
            conv_separate_features = ('cnvsep' in cost_model_type)
            count_sum = ('sum' in cost_model_type)
            cost_model = NASBench201LinearCostModel(
                config_keys=CONFIG_KEYS,
                map_config_values=map_config_values,
                conv_separate_features=conv_separate_features,
                count_sum=count_sum)
        else:
            from syne_tune.optimizer.schedulers.searchers.bayesopt.models.cost.sklearn_cost_model \
                import ScikitLearnCostModel

            cost_model = ScikitLearnCostModel(cost_model_type)
        return cost_model
    except Exception:
        return None
