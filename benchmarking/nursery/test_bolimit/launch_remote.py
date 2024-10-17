from pathlib import Path

from benchmark_definitions import benchmark_definitions
from baselines import methods
from syne_tune.experiments.launchers.launch_remote_simulator import launch_remote


if __name__ == "__main__":

    def _is_expensive_method(method: str) -> bool:
        return method.startswith("BO")

    entry_point = Path(__file__).parent / "hpo_main.py"
    launch_remote(
        entry_point=entry_point,
        methods=methods,
        benchmark_definitions=benchmark_definitions,
        is_expensive_method=_is_expensive_method,
    )
