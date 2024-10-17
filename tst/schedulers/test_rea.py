import pytest

from syne_tune.config_space import randint, choice
from syne_tune.optimizer.schedulers.searchers.regularized_evolution import (
    RegularizedEvolution,
)


def test_rea_config_space_size_one():
    config_space = {
        "a": randint(lower=1, upper=1),
        "b": choice(["a"]),
        "c": 25,
        "d": "dummy",
    }
    with pytest.raises(AssertionError):
        searcher = RegularizedEvolution(
            config_space=config_space,
            metric="error",
            mode="min",
            random_seed=314159,
        )
