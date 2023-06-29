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

from syne_tune.experiments.benchmark_definitions.common import RealBenchmarkDefinition
from benchmarking.training_scripts.transformer_wikitext2.training_script import (
    _config_space,
    METRIC_NAME,
    RESOURCE_ATTR,
    MAX_RESOURCE_ATTR,
)
from syne_tune.remote.constants import (
    DEFAULT_GPU_INSTANCE_1GPU,
    DEFAULT_GPU_INSTANCE_4GPU,
)


def transformer_wikitext2_benchmark(sagemaker_backend: bool = False, **kwargs):
    if sagemaker_backend:
        instance_type = DEFAULT_GPU_INSTANCE_1GPU
    else:
        # For local backend, GPU cores serve different workers
        instance_type = DEFAULT_GPU_INSTANCE_4GPU
    fixed_parameters = dict(
        **{MAX_RESOURCE_ATTR: 40},
        d_model=256,
        ffn_ratio=1,
        nlayers=2,
        nhead=2,
        bptt=35,
        optimizer_name="sgd",
        input_data_dir="./",
        use_cuda=1,
        seed=1111,
        precision="float",
        log_interval=200,
    )
    config_space = {**_config_space, **fixed_parameters}
    _kwargs = dict(
        script=Path(__file__).parent.parent
        / "training_scripts"
        / "transformer_wikitext2"
        / "training_script.py",
        config_space=config_space,
        metric=METRIC_NAME,
        mode="min",
        max_resource_attr=MAX_RESOURCE_ATTR,
        resource_attr=RESOURCE_ATTR,
        max_wallclock_time=5 * 3600,
        n_workers=4,
        instance_type=instance_type,
        framework="PyTorch",
    )
    _kwargs.update(kwargs)
    return RealBenchmarkDefinition(**_kwargs)


# Support for cost models:
#
# from benchmarking.utils import get_cost_model_for_batch_size
# from benchmarking.training_scripts.transformer_wikitext2.training_script import (
#     BATCH_SIZE_LOWER,
#     BATCH_SIZE_UPPER,
#     BATCH_SIZE_KEY,
# )
# cost_model = get_cost_model_for_batch_size(
#     cost_model_type="quadratic_spline",
#     batch_size_key = BATCH_SIZE_KEY,
#     batch_size_range = (BATCH_SIZE_LOWER, BATCH_SIZE_UPPER),
# )
