import pytest

import numpy as np

from syne_tune.config_space import randint, choice
from syne_tune.optimizer.schedulers.searchers.regularized_evolution import (
    RegularizedEvolution, mutate_config, sample_random_config
)
from syne_tune.optimizer.schedulers.multiobjective.multi_objective_regularized_evolution import MultiObjectiveRegularizedEvolution


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
            random_seed=314159,
        )

def test_random_config():
    config_space = {
        "a": randint(lower=1, upper=5),
        "b": 25
    }

    config = sample_random_config(config_space, rng=np.random.RandomState(42))
    assert "a" in config
    assert 1 <= config['a'] <= 5
    assert "b" in config
    assert config['b'] == 25


def test_mutate_config():
    config_space = {
        "a": randint(lower=1, upper=5),
        "b": 25
    }

    config = {"a": 1, "b": 25}
    mutated_config = mutate_config(config, config_space, rng=np.random.RandomState(42))

    assert config['b'] == mutated_config['b']
    assert config['a'] != mutated_config['a']


def test_rea_population():

    config_space = {
        "a": randint(lower=1, upper=100),
        "b": choice(["a", "b", "c", "d"]),
    }

    pop_size = 5

    searcher = RegularizedEvolution(
        config_space=config_space,
        random_seed=314159,
        population_size=pop_size
    )

    history = []
    for i in range(pop_size):
        config = searcher.suggest()
        searcher.on_trial_result(i, config, np.random.rand())
        history.append(config)

    assert len(searcher.population) == 5
    config = searcher.suggest()
    searcher.on_trial_result(pop_size + 1, config, np.random.rand())

    # assert that we removed the oldest element from the population
    assert history[0] not in searcher.population


def test_mo_rea_population():

    config_space = {
        "a": randint(lower=1, upper=100),
        "b": choice(["a", "b", "c", "d"]),
    }

    pop_size = 5

    searcher = MultiObjectiveRegularizedEvolution(
        config_space=config_space,
        random_seed=314159,
        population_size=pop_size
    )

    history = []
    for i in range(pop_size):
        config = searcher.suggest()
        searcher.on_trial_result(i, config, [np.random.rand(), np.random.rand()])
        history.append(config)

    assert len(searcher.population) == 5
    config = searcher.suggest()
    searcher.on_trial_result(pop_size + 1, config, [np.random.rand(), np.random.rand()])

    # assert that we removed the oldest element from the population
    assert history[0] not in searcher.population
