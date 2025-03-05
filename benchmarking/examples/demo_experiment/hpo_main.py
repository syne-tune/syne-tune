from typing import Optional, Dict, Any, List

from baselines import methods
from benchmark_definitions import benchmark_definitions
from syne_tune import Tuner
from syne_tune.experiments.launchers.hpo_main_simulator import main
from syne_tune.optimizer.schedulers import LegacyHyperbandScheduler
from syne_tune.results_callback import ExtraResultsComposer


RESOURCE_LEVELS = [1, 3, 9, 27, 81]


class RungLevelsExtraResults(ExtraResultsComposer):
    """
    We would like to monitor the sizes of rung levels over time. This is an extra
    information normally not recorded and stored.
    """

    def __call__(self, tuner: Tuner) -> Optional[Dict[str, Any]]:
        if not isinstance(tuner.scheduler, LegacyHyperbandScheduler):
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
