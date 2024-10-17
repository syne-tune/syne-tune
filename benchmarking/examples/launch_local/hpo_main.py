from benchmarking.examples.launch_local.baselines import methods
from benchmarking.benchmark_definitions import (
    real_benchmark_definitions as benchmark_definitions,
)
from syne_tune.experiments.launchers.hpo_main_local import main


if __name__ == "__main__":
    main(methods, benchmark_definitions)
