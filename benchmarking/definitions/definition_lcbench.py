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
from benchmarking.blackbox_repository.conversion_scripts.scripts.lcbench.lcbench \
    import METRIC_ELAPSED_TIME, METRIC_ACCURACY, RESOURCE_ATTR, \
    BLACKBOX_NAME, MAX_RESOURCE_LEVEL, CONFIGURATION_SPACE


DATASET_NAMES = [
    "APSFailure",
    "Amazon_employee_access",
    "Australian",
    "Fashion-MNIST",
    "KDDCup09_appetency",
    "MiniBooNE",
    "adult",
    "airlines",
    "albert",
    "bank-marketing",
    "blood-transfusion-service-center",
    "car",
    "christine",
    "cnae-9",
    "connect-4",
    "covertype",
    "credit-g",
    "dionis",
    "fabert",
    "helena",
    "higgs",
    "jannis",
    "jasmine",
    "jungle_chess_2pcs_raw_endgame_complete",
    "kc1",
    "kr-vs-kp",
    "mfeat-factors",
    "nomao",
    "numerai28.6",
    "phoneme",
    "segment",
    "shuttle",
    "sylvine",
    "vehicle",
    "volkert",
]


def lcbench_default_params(params=None):
    return {
        'max_resource_level': MAX_RESOURCE_LEVEL,
        'grace_period': 1,
        'reduction_factor': 3,
        'instance_type': 'ml.m5.large',
        'num_workers': 4,
        'framework': 'PyTorch',
        'framework_version': '1.6',
        'dataset_name': 'Fashion-MNIST',
    }


def lcbench_benchmark(params):
    """
    The underlying tabulated blackbox does not have an `elapsed_time_attr`,
    but only a `time_this_resource_attr`.

    """
    config_space = dict(
        CONFIGURATION_SPACE,
        epochs=params['max_resource_level'],
        dataset_name=params['dataset_name'])
    return {
        'script': None,
        'metric': METRIC_ACCURACY,
        'mode': 'max',
        'resource_attr': RESOURCE_ATTR,
        'elapsed_time_attr': METRIC_ELAPSED_TIME,
        'max_resource_attr': 'epochs',
        'config_space': config_space,
        'cost_model': None,
        'supports_simulated': True,
        'blackbox_name': BLACKBOX_NAME,
    }
