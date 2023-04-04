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
from typing import Dict, Any, Optional, List

from benchmarking.commons.hpo_main_simulator import main
from benchmarking.nursery.benchmark_dyhpo.baselines import methods
from benchmarking.nursery.benchmark_dyhpo.benchmark_definitions import (
    benchmark_definitions,
)
from syne_tune import Tuner
from syne_tune.optimizer.schedulers import HyperbandScheduler
from syne_tune.optimizer.schedulers.searchers.dyhpo.hyperband_dyhpo import (
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


def map_extra_args(args, method: str, method_kwargs: Dict[str, Any]) -> Dict[str, Any]:
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
            isinstance(scheduler, HyperbandScheduler)
            and scheduler.scheduler_type == "dyhpo"
        ):
            result = scheduler.terminator._rung_systems[0].summary_schedule_records()
        return result

    def keys(self) -> List[str]:
        return DyHPORungSystem.summary_schedule_keys()


if __name__ == "__main__":
    extra_results = DyHPOExtraResults()
    main(methods, benchmark_definitions, extra_args, map_extra_args, extra_results)
