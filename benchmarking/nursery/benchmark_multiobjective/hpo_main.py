
from benchmarking.nursery.benchmark_multiobjective.baselines import methods
from benchmarking.nursery.benchmark_multiobjective.benchmark_definitions import (
    benchmark_definitions,
)
from syne_tune.experiments.launchers.hpo_main_simulator import main

if __name__ == "__main__":
    main(methods, benchmark_definitions)
