from typing import Dict, Any

from benchmarking.examples.fine_tuning_transformer_glue.baselines import methods
from benchmarking.benchmark_definitions import (
    real_benchmark_definitions as benchmark_definitions,
)
from benchmarking.benchmark_definitions.finetune_transformer_glue import (
    PRETRAINED_MODELS,
    MAX_RESOURCE_ATTR,
    MODEL_TYPE_ATTR,
)
from syne_tune.config_space import Domain
from syne_tune.experiments.launchers.hpo_main_local import main


extra_args = [
    dict(
        name="num_train_epochs",
        type=int,
        default=3,
        help="Maximum number of training epochs",
    ),
    dict(
        name="model_type",
        type=str,
        default="bert-base-cased",
        choices=tuple(PRETRAINED_MODELS),
        help="Pre-trained model to start fine-tuning from",
    ),
]


def map_method_args(args, method: str, method_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    # We need to change ``method_kwargs.config_space``, based on ``extra_args``
    new_method_kwargs = method_kwargs.copy()
    new_config_space = new_method_kwargs["config_space"].copy()
    choose_model = isinstance(["model_name_or_path"], Domain)
    new_config_space[MAX_RESOURCE_ATTR] = args.num_train_epochs
    if not choose_model:
        new_config_space[MODEL_TYPE_ATTR] = args.model_type
    else:
        # Need to change ``points_to_evaluate``
        default_configuration = new_method_kwargs["scheduler_kwargs"][
            "points_to_evaluate"
        ][0]
        new_default_configuration = {
            **default_configuration,
            MODEL_TYPE_ATTR: args.model_type,
        }
        new_method_kwargs["scheduler_kwargs"]["points_to_evaluate"] = [
            new_default_configuration
        ]
    new_method_kwargs["config_space"] = new_config_space
    return new_method_kwargs


if __name__ == "__main__":
    main(methods, benchmark_definitions, extra_args, map_method_args)
