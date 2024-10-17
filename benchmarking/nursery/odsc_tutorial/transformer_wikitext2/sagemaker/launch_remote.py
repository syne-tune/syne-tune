from pathlib import Path

from transformer_wikitext2.baselines import methods
from transformer_wikitext2.benchmark_definitions import benchmark_definitions
from syne_tune.experiments.launchers.launch_remote_sagemaker import launch_remote


if __name__ == "__main__":
    entry_point = Path(__file__).parent / "hpo_main.py"
    source_dependencies = [str(Path(__file__).parent.parent)]
    launch_remote(
        entry_point=entry_point,
        methods=methods,
        benchmark_definitions=benchmark_definitions,
        source_dependencies=source_dependencies,
    )
