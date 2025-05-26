from typing import Optional, Any

from benchmarking.benchmark_definitions import (
    real_benchmark_definitions as benchmark_definitions,
)
from benchmarking.nursery.test_checkpoints.baselines import methods
from syne_tune import Tuner
from syne_tune.callbacks.hyperband_remove_checkpoints_callback import (
    HyperbandRemoveCheckpointsCommon,
)
from syne_tune.experiments.launchers.hpo_main_local import main
from syne_tune.results_callback import ExtraResultsComposer
from syne_tune.util import find_first_of_type


class CPRemovalExtraResults(ExtraResultsComposer):
    def __call__(self, tuner: Tuner) -> Optional[dict[str, Any]]:
        callback = find_first_of_type(tuner.callbacks, HyperbandRemoveCheckpointsCommon)
        return None if callback is None else callback.extra_results()

    def keys(self) -> list[str]:
        return HyperbandRemoveCheckpointsCommon.extra_results_keys()


if __name__ == "__main__":
    extra_results = CPRemovalExtraResults()
    main(methods, benchmark_definitions, extra_results=extra_results)
