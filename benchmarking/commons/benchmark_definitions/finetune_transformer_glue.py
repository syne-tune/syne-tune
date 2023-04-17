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
from typing import Dict, Optional
from pathlib import Path

from benchmarking.commons.benchmark_definitions.common import RealBenchmarkDefinition
from syne_tune.config_space import uniform, loguniform, choice, randint
from syne_tune.remote.estimators import (
    DEFAULT_GPU_INSTANCE_1GPU,
    DEFAULT_GPU_INSTANCE_4GPU,
)


# Different GLUE tasks and their metric names
# TODO: Adjust default values of ``max_wallclock_time``
TASK2METRICSMODE = {
    "cola": {
        "metric": "matthews_correlation",
        "mode": "max",
        "max_wallclock_time": 2700,
    },
    "mnli": {"metric": "accuracy", "mode": "max", "max_wallclock_time": 2700},
    "mrpc": {"metric": "f1", "mode": "max", "max_wallclock_time": 2700},
    "qnli": {"metric": "accuracy", "mode": "max", "max_wallclock_time": 2700},
    "qqp": {"metric": "f1", "mode": "max", "max_wallclock_time": 2700},
    "rte": {"metric": "accuracy", "mode": "max", "max_wallclock_time": 2700},
    "sst2": {"metric": "accuracy", "mode": "max", "max_wallclock_time": 2700},
    "stsb": {"metric": "spearmanr", "mode": "max", "max_wallclock_time": 2700},
    "wnli": {"metric": "accuracy", "mode": "max", "max_wallclock_time": 2700},
}


# Pre-trained models from HuggingFace zoo considered here
PRETRAINED_MODELS = [
    "bert-base-cased",
    "bert-base-uncased",
    "distilbert-base-uncased",
    "distilbert-base-cased",
    "roberta-base",
    "albert-base-v2",
    "distilroberta-base",
    "xlnet-base-cased",
    "albert-base-v1",
]


RESOURCE_ATTR = "epoch"
MAX_RESOURCE_ATTR = "num_train_epochs"
MODEL_TYPE_ATTR = "model_name_or_path"


def finetune_transformer_glue_benchmark(
    sagemaker_backend: bool = False,
    choose_model: bool = False,
    dataset: str = "rte",
    model_type: str = "bert-base-cased",
    max_wallclock_time: Optional[int] = None,
    n_workers: int = 4,
    num_train_epochs: int = 3,
    train_valid_fraction: float = 0.7,
    random_seed: int = 31415927,
) -> RealBenchmarkDefinition:
    """
    This benchmark consists of fine-tuning a Hugging Face transformer model,
    selected from the zoo, on one of the GLUE benchmarks:

        | Wang etal.
        | GLUE: A Multi-task Benchmark and Analysis Platform for Natural
        | Language Understanding
        | ICLR 2019

    :param sagemaker_backend: Use SageMaker backend? This affects the choice
        of instance type. Defaults to ``False``
    :param choose_model: Should tuning involve selecting the best pre-trained
        model from ``PRETRAINED_MODELS``? If so, the configuration space is
        extended by another choice variable. Defaults to ``False``
    :param dataset: Name of GLUE task, from ``TASK2METRICSMODE``. Defaults to
        "rte"
    :param model_type: Pre-trained model to be used. If ``choose_model`` is
        set, this is the model used in the first evaluation. Defaults to
        "bert-base-cased"
    :param max_wallclock_time: Maximum wall-clock time in secs. Defaults to 2700
    :param n_workers: Number of workers. Defaults to 4
    :param num_train_epochs: Maximum number of epochs for fine-tuning. Defaults
        to 3
    :param train_valid_fraction: The original training set is split into training
        and validation part, this is the fraction of the training part
    :param random_seed: Random seed for training script
    """
    if sagemaker_backend:
        instance_type = DEFAULT_GPU_INSTANCE_1GPU
    else:
        # For local backend, GPU cores serve different workers, so we
        # need more memory
        instance_type = DEFAULT_GPU_INSTANCE_4GPU

    task_defaults = TASK2METRICSMODE[dataset]
    metric = "eval_" + task_defaults["metric"]
    mode = task_defaults["mode"]
    if max_wallclock_time is None:
        max_wallclock_time = task_defaults["max_wallclock_time"]

    hyperparameter_space = {
        "learning_rate": loguniform(1e-6, 1e-4),
        "per_device_train_batch_size": randint(16, 48),
        "warmup_ratio": uniform(0, 0.5),
    }
    # We use ``save_strategy="no"`` in order to save disk space. This must be
    # changed to "epoch" for HPO methods which require checkpointing.
    fixed_parameters = {
        MAX_RESOURCE_ATTR: num_train_epochs,
        MODEL_TYPE_ATTR: model_type,
        "task_name": dataset,
        "train_valid_fraction": train_valid_fraction,
        "seed": random_seed,
        "do_train": True,
        "max_seq_length": 128,
        "output_dir": "tmp/" + dataset,
        "evaluation_strategy": "epoch",
        "save_strategy": "no",  # change to "epoch" if checkpoints are needed!
        "save_total_limit": 1,
    }

    config_space = {**hyperparameter_space, **fixed_parameters}
    # This is the default configuration provided by Hugging Face. It will always
    # be evaluated first
    default_configuration = {
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 32,
        "warmup_ratio": 0.0,
    }

    # Combine HPO with model selection:
    # Just another categorical hyperparameter
    if choose_model:
        config_space[MODEL_TYPE_ATTR] = choice(PRETRAINED_MODELS)
        default_configuration[MODEL_TYPE_ATTR] = model_type

    kwargs = dict(
        script=Path(__file__).parent.parent.parent
        / "training_scripts"
        / "finetune_transformer_glue"
        / "run_glue_modified.py",
        config_space=config_space,
        max_wallclock_time=max_wallclock_time,
        n_workers=n_workers,
        instance_type=instance_type,
        metric=metric,
        mode=mode,
        max_resource_attr=MAX_RESOURCE_ATTR,
        resource_attr=RESOURCE_ATTR,
        framework="PyTorch",
        points_to_evaluate=[default_configuration],
    )
    return RealBenchmarkDefinition(**kwargs)


def finetune_transformer_glue_all_benchmarks(
    sagemaker_backend: bool = False,
    model_type: str = "bert-base-cased",
    max_wallclock_time: Optional[int] = None,
    n_workers: int = 4,
    num_train_epochs: int = 3,
    train_valid_fraction: float = 0.7,
    random_seed: int = 31415927,
) -> Dict[str, RealBenchmarkDefinition]:
    result = dict()
    for choose_model in [True, False]:
        prefix = "finetune_transformer_glue_"
        if choose_model:
            prefix += "modsel_"
        result.update(
            {
                prefix
                + dataset: finetune_transformer_glue_benchmark(
                    sagemaker_backend=sagemaker_backend,
                    choose_model=choose_model,
                    dataset=dataset,
                    model_type=model_type,
                    max_wallclock_time=max_wallclock_time,
                    n_workers=n_workers,
                    num_train_epochs=num_train_epochs,
                    train_valid_fraction=train_valid_fraction,
                    random_seed=random_seed,
                )
                for dataset in TASK2METRICSMODE.keys()
            }
        )
    return result
