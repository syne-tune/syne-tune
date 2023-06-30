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

from benchmarking.training_scripts.lstm_wikitext2.lstm_wikitext2 import (
    _config_space,
    METRIC_NAME,
    RESOURCE_ATTR,
)
from syne_tune.experiments.benchmark_definitions.common import RealBenchmarkDefinition
from syne_tune.remote.constants import (
    DEFAULT_GPU_INSTANCE_1GPU,
    DEFAULT_GPU_INSTANCE_4GPU,
)


def lstm_wikitext2_benchmark(sagemaker_backend: bool = False, **kwargs):
    if sagemaker_backend:
        instance_type = DEFAULT_GPU_INSTANCE_1GPU
        n_workers = 8
    else:
        # For local backend, GPU cores serve different workers, so we
        # need more memory
        instance_type = DEFAULT_GPU_INSTANCE_4GPU
        n_workers = 4
    config_space = dict(
        _config_space,
        dataset_path="./",
        epochs=81,
        report_current_best="False",
    )
    _kwargs = dict(
        script=Path(__file__).parent.parent
        / "training_scripts"
        / "lstm_wikitext2"
        / "lstm_wikitext2.py",
        config_space=config_space,
        max_wallclock_time=7 * 3600,
        n_workers=n_workers,
        instance_type=instance_type,
        metric=METRIC_NAME,
        mode="max",
        max_resource_attr="epochs",
        resource_attr=RESOURCE_ATTR,
        framework="PyTorch",
    )
    _kwargs.update(kwargs)
    return RealBenchmarkDefinition(**_kwargs)


# Support for cost models:
#
# from benchmarking.utils import get_cost_model_for_batch_size
# from benchmarking.training_scripts.lstm_wikitext2.lstm_wikitext2 import (
#     BATCH_SIZE_LOWER,
#     BATCH_SIZE_UPPER,
#     BATCH_SIZE_KEY,
# )
# cost_model = get_cost_model_for_batch_size(
#     cost_model_type="quadratic_spline",
#     batch_size_key = BATCH_SIZE_KEY,
#     batch_size_range = (BATCH_SIZE_LOWER, BATCH_SIZE_UPPER),
# )
