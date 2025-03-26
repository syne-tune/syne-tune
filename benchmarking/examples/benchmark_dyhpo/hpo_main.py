from typing import Dict, Any, Optional, List

from baselines import methods
from benchmark_definitions import benchmark_definitions
from syne_tune import Tuner
from syne_tune.experiments.launchers.hpo_main_simulator import main
from syne_tune.optimizer.schedulers import LegacyHyperbandScheduler
from syne_tune.optimizer.schedulers.searchers.legacy_dyhpo.hyperband_dyhpo import (
    DyHPORungSystem,
)
from syne_tune.results_callback import ExtraResultsComposer
from syne_tune.util import recursive_merge


extra_args = [
    dict(
        name="num_brackets",
        type=int,
        help="Number of brackets",
    ),
    dict(
        name="probability_sh",
        type=float,
        help="Parameter for DyHPO: Probability of making SH promotion decision",
    ),
    dict(
        name="rung_increment",
        type=int,
        help="Increment between rung levels",
    ),
    dict(
        name="opt_skip_period",
        type=int,
        help="Period for fitting surrogate model. Only used for DyHPO",
    ),
]


def map_method_args(args, method: str, method_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    scheduler_kwargs = dict()
    if method.startswith("DYHPO"):
        if args.rung_increment is not None:
            scheduler_kwargs["rung_increment"] = args.rung_increment
        if args.probability_sh is not None:
            scheduler_kwargs["probability_sh"] = args.probability_sh
        if args.opt_skip_period is not None:
            scheduler_kwargs["search_options"] = {
                "opt_skip_period": args.opt_skip_period,
            }
    if args.num_brackets is not None:
        scheduler_kwargs["brackets"] = args.num_brackets
    if scheduler_kwargs:
        method_kwargs = recursive_merge(
            method_kwargs, {"scheduler_kwargs": scheduler_kwargs}
        )
    return method_kwargs


class DyHPOExtraResults(ExtraResultsComposer):
    def __call__(self, tuner: "Tuner") -> Optional[Dict[str, Any]]:
        # Only for DyHPO
        result = None
        scheduler = tuner.scheduler
        if (
            isinstance(scheduler, LegacyHyperbandScheduler)
            and scheduler.scheduler_type == "legacy_dyhpo"
        ):
            result = scheduler.terminator._rung_systems[0].summary_schedule_records()
        return result

    def keys(self) -> List[str]:
        return DyHPORungSystem.summary_schedule_keys()


if __name__ == "__main__":
    extra_results = DyHPOExtraResults()
    main(methods, benchmark_definitions, extra_args, map_method_args, extra_results)
