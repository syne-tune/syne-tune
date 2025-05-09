from pathlib import Path

import benchmarking
from benchmarking.benchmark_definitions import (
    real_benchmark_definitions as benchmark_definitions,
)
from benchmarking.examples.fine_tuning_transformer_swag.baselines import methods
from benchmarking.examples.fine_tuning_transformer_swag.hpo_main import extra_args
from syne_tune.experiments.launchers.launch_remote_local import launch_remote


if __name__ == "__main__":
    entry_point = Path(__file__).parent / "hpo_main.py"
    launch_remote(
        entry_point=entry_point,
        methods=methods,
        benchmark_definitions=benchmark_definitions,
        source_dependencies=benchmarking.__path__,
        extra_args=extra_args,
    )
