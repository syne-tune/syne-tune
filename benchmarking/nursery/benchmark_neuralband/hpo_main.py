from typing import Dict, Any
from syne_tune.experiments.launchers.hpo_main_simulator import main
from baselines import methods
from benchmark_definitions import benchmark_definitions
from syne_tune.util import recursive_merge


extra_args = [
    dict(
        name="num_brackets",
        type=int,
        help="Number of brackets",
    ),
]


def map_method_args(args, method: str, method_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if args.num_brackets is not None:
        new_dict = {
            "scheduler_kwargs": {"brackets": args.num_brackets},
        }
        method_kwargs = recursive_merge(method_kwargs, new_dict)
    return method_kwargs


if __name__ == "__main__":
    main(methods, benchmark_definitions, extra_args, map_method_args)
