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

from benchmarking.commons.benchmark_definitions.common import RealBenchmarkDefinition
from syne_tune.config_space import uniform, loguniform
from syne_tune.remote.estimators import (
    DEFAULT_GPU_INSTANCE_1GPU,
    DEFAULT_GPU_INSTANCE_4GPU,
)

METRIC = "eval_accuracy"
MODE = "max"
RESOURCE_ATTR = "epoch"
MAX_RESOURCE_ATTR = "epochs"
BATCH_SIZE_ATTR = "per_device_train_batch_size"


def finetune_transformer_swag_benchmark(
    sagemaker_backend: bool = False,
    max_wallclock_time: int = 5 * 3600,
    n_workers: int = 4,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
) -> RealBenchmarkDefinition:
    """
    :param sagemaker_backend: Use SageMaker backend? This affects the choice
        of instance type. Defaults to ``False``
    :param max_wallclock_time: Maximum wall-clock time in secs. Defaults to 1800
    :param n_workers: Number of workers. Defaults to 4
    :param num_train_epochs: Maximum number of epochs for fine-tuning. Defaults
        to 3
    :param per_device_train_batch_size: Batch size per device. Defaults to 8
    """
    if sagemaker_backend:
        instance_type = DEFAULT_GPU_INSTANCE_1GPU
    else:
        # For local backend, GPU cores serve different workers, so we
        # need more memory
        instance_type = DEFAULT_GPU_INSTANCE_4GPU

    config_space = {
        "learning_rate": loguniform(1e-6, 1e-4),
        "warmup_ratio": uniform(0, 0.5),
        "weight_decay": uniform(0, 1e-1),
        "adam_beta1": uniform(0.0, 0.9999),
        "adam_beta2": uniform(0.0, 0.9999),
        "adam_epsilon": loguniform(1e-10, 1e-6),
        "max_grad_norm": uniform(0, 2),
        BATCH_SIZE_ATTR: per_device_train_batch_size,
        MAX_RESOURCE_ATTR: num_train_epochs,
    }

    default_configuration = {
        "learning_rate": 5e-5,
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
    }

    kwargs = dict(
        script=Path(__file__).parent.parent.parent
        / "training_scripts"
        / "finetune_transformer_swag"
        / "multiple_choice_on_swag.py",
        config_space=config_space,
        max_wallclock_time=max_wallclock_time,
        n_workers=n_workers,
        instance_type=instance_type,
        metric=METRIC,
        mode=MODE,
        max_resource_attr=MAX_RESOURCE_ATTR,
        resource_attr=RESOURCE_ATTR,
        framework="PyTorch",
        points_to_evaluate=[default_configuration],
    )
    return RealBenchmarkDefinition(**kwargs)
