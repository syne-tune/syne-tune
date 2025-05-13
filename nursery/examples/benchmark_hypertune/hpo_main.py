from typing import Dict, Any

from baselines import methods
from benchmark_definitions import benchmark_definitions
from syne_tune.experiments.launchers.hpo_main_simulator import main
from syne_tune.util import recursive_merge


extra_args = [
    dict(
        name="num_brackets",
        type=int,
        help="Number of brackets",
    ),
    dict(
        name="num_samples",
        type=int,
        default=50,
        help="Number of samples for Hyper-Tune distribution",
    ),
]


def map_method_args(args, method: str, method_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if method.startswith("HYPERTUNE"):
        scheduler_kwargs = {
            "search_options": {"hypertune_distribution_num_samples": args.num_samples},
        }
    else:
        scheduler_kwargs = dict()
    if args.num_brackets is not None:
        scheduler_kwargs["brackets"] = args.num_brackets
    if scheduler_kwargs:
        method_kwargs = recursive_merge(
            method_kwargs, {"scheduler_kwargs": scheduler_kwargs}
        )
    return method_kwargs


if __name__ == "__main__":
    main(methods, benchmark_definitions, extra_args, map_method_args)
