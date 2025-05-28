from typing import Any

from benchmarking.examples.fine_tuning_transformer_swag.baselines import methods
from benchmarking.benchmark_definitions import (
    real_benchmark_definitions as benchmark_definitions,
)
from benchmarking.benchmark_definitions.finetune_transformer_swag import (
    MAX_RESOURCE_ATTR,
    BATCH_SIZE_ATTR,
)
from syne_tune.experiments.launchers.hpo_main_local import main


extra_args = [
    dict(
        name="num_train_epochs",
        type=int,
        default=3,
        help="Maximum number of training epochs",
    ),
    dict(
        name="batch_size",
        type=int,
        default=8,
        help="Training batch size (per device)",
    ),
]


def map_method_args(args, method: str, method_kwargs: dict[str, Any]) -> dict[str, Any]:
    # We need to change ``method_kwargs.config_space``, based on ``extra_args``
    new_method_kwargs = method_kwargs.copy()
    new_config_space = new_method_kwargs["config_space"].copy()
    new_config_space[MAX_RESOURCE_ATTR] = args.num_train_epochs
    new_config_space[BATCH_SIZE_ATTR] = args.batch_size
    new_method_kwargs["config_space"] = new_config_space
    return new_method_kwargs


if __name__ == "__main__":
    main(methods, benchmark_definitions, extra_args, map_method_args)
