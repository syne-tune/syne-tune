from pathlib import Path

import benchmarking
from benchmarking.benchmark_definitions import (
    real_benchmark_definitions as benchmark_definitions,
)
from benchmarking.examples.launch_local.baselines import methods
from syne_tune.experiments.launchers.launch_remote_local import launch_remote


if __name__ == "__main__":
    entry_point = Path(__file__).parent / "hpo_main.py"
    launch_remote(
        entry_point=entry_point,
        methods=methods,
        benchmark_definitions=benchmark_definitions,
        source_dependencies=benchmarking.__path__,
    )
