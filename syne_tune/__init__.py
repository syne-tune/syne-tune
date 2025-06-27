from pathlib import Path

try:
    # The reason for conditional imports is that ``read_version`` is called
    # by ``setup.py`` before any dependencies are installed
    from syne_tune.stopping_criterion import StoppingCriterion  # noqa: F401
    from syne_tune.report import Reporter  # noqa: F401
    from syne_tune.tuner import Tuner  # noqa: F401

    __all__ = ["StoppingCriterion", "Tuner", "Reporter"]
except ImportError:
    __all__ = []


def read_version():
    with open(Path(__file__).parent / "version", "r") as f:
        return f.readline().strip().replace('"', "")


__version__ = read_version()
