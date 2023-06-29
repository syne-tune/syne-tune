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

from benchmarking.training_scripts.distilbert_on_imdb.distilbert_on_imdb import (
    METRIC_ACCURACY,
    RESOURCE_ATTR,
    _config_space,
)
from syne_tune.experiments.benchmark_definitions.common import RealBenchmarkDefinition
from syne_tune.remote.constants import DEFAULT_GPU_INSTANCE_1GPU


def distilbert_imdb_benchmark(sagemaker_backend: bool = False, **kwargs):
    config_space = dict(
        _config_space,
        dataset_path="./",
        epochs=15,
    )
    _kwargs = dict(
        script=Path(__file__).parent.parent
        / "training_scripts"
        / "distilbert_on_imdb"
        / "distilbert_on_imdb.py",
        config_space=config_space,
        max_wallclock_time=3 * 3600,  # TODO
        n_workers=4,
        instance_type=DEFAULT_GPU_INSTANCE_1GPU,
        metric=METRIC_ACCURACY,
        mode="max",
        max_resource_attr="epochs",
        resource_attr=RESOURCE_ATTR,
        framework="HuggingFace",
    )
    _kwargs.update(kwargs)
    return RealBenchmarkDefinition(**_kwargs)
