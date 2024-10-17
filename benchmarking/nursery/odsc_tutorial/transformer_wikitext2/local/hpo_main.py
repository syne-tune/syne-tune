from transformer_wikitext2.baselines import methods
from transformer_wikitext2.benchmark_definitions import benchmark_definitions
from syne_tune.experiments.launchers.hpo_main_local import main


if __name__ == "__main__":
    main(methods, benchmark_definitions)
