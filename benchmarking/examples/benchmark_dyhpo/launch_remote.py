from pathlib import Path

from benchmark_definitions import benchmark_definitions
from baselines import methods
from hpo_main import extra_args
from syne_tune.experiments.launchers.launch_remote_simulator import launch_remote


if __name__ == "__main__":

    def _is_expensive_method(method: str) -> bool:
        return method not in ["RS", "BO", "ASHA"]

    entry_point = Path(__file__).parent / "hpo_main.py"
    launch_remote(
        entry_point=entry_point,
        methods=methods,
        benchmark_definitions=benchmark_definitions,
        extra_args=extra_args,
        is_expensive_method=_is_expensive_method,
    )
