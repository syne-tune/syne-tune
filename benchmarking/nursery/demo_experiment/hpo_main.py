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
from typing import Optional, Dict, Any, List

from benchmarking.commons.hpo_main_simulator import main
from benchmarking.nursery.demo_experiment.baselines import methods
from benchmarking.nursery.demo_experiment.benchmark_definitions import (
    benchmark_definitions,
)
from syne_tune import Tuner
from syne_tune.optimizer.schedulers import HyperbandScheduler
from syne_tune.results_callback import ExtraResultsComposer


RESOURCE_LEVELS = [1, 3, 9, 27, 81]


class RungLevelsExtraResults(ExtraResultsComposer):
    """
    We would like to monitor the sizes of rung levels over time. This is an extra
    information normally not recorded and stored.
    """

    def __call__(self, tuner: Tuner) -> Optional[Dict[str, Any]]:
        if not isinstance(tuner.scheduler, HyperbandScheduler):
            return None
        rung_information = tuner.scheduler.terminator.information_for_rungs()
        return {
            f"num_at_level{resource}": num_entries
            for resource, num_entries, _ in rung_information
            if resource in RESOURCE_LEVELS
        }

    def keys(self) -> List[str]:
        return [f"num_at_level{r}" for r in RESOURCE_LEVELS]


if __name__ == "__main__":
    extra_results = RungLevelsExtraResults()
    main(methods, benchmark_definitions, extra_results=extra_results)
