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
DistilBERT fine-tuned on IMDB sentiment classification task
"""
from pathlib import Path

from benchmarking.training_scripts.distilbert_on_imdb.distilbert_on_imdb import \
    METRIC_ACCURACY, RESOURCE_ATTR, _config_space


def distilbert_imdb_default_params(params=None):
    return {
        'max_resource_level': 15,
        'instance_type': 'ml.g4dn.xlarge',
        'num_workers': 4,
        'framework': 'HuggingFace',
        'framework_version': '4.4',
        'pytorch_version': '1.6',
        'dataset_path': './'
    }


def distilbert_imdb_benchmark(params):
    config_space = dict(
        _config_space,
        dataset_path=params['dataset_path'],
        max_steps=params['max_resource_level'])
    return {
        'script':  Path(__file__).parent.parent / "training_scripts" / "distilbert_on_imdb" / "distilbert_on_imdb.py",
        'metric': METRIC_ACCURACY,
        'mode': 'max',
        'resource_attr': RESOURCE_ATTR,
        'max_resource_attr': 'epochs',
        'config_space': config_space,
    }
