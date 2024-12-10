import copy
import logging

import numpy as np

from collections import deque
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from syne_tune.config_space import config_space_size, non_constant_hyperparameter_keys
from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import (
    SingleObjectiveBaseSearcher,
)


logger = logging.getLogger(__name__)


@dataclass
class PopulationElement:
    score: float = 0
    config: Dict[str, Any] = None
    results: List[float] = None


def mutate_config(
    config: Dict[str, Any],
    config_space: Dict[str, Any],
    rng: np.random.RandomState,
    num_try: int = 1000,
) -> Dict[str, Any]:

    child_config = copy.deepcopy(config)
    hp_name = rng.choice(non_constant_hyperparameter_keys(config_space))

    for i in range(num_try):
        child_config[hp_name] = config_space[hp_name].sample(random_state=rng)

        # make sure that we actually sample a new value
        if child_config[hp_name] != config[hp_name]:
            break

    return child_config


def sample_random_config(
    config_space: Dict[str, Any], rng: np.random.RandomState
) -> Dict[str, Any]:
    return {
        k: v.sample() if hasattr(v, "sample") else v for k, v in config_space.items()
    }


class RegularizedEvolution(SingleObjectiveBaseSearcher):
    """
    Implements the regularized evolution algorithm. The original implementation
    only considers categorical hyperparameters. For integer and float parameters
    we sample a new value uniformly at random. Reference:

        | Real, E., Aggarwal, A., Huang, Y., and Le, Q. V.
        | Regularized Evolution for Image Classifier Architecture Search.
        | In Proceedings of the Conference on Artificial Intelligence (AAAI’19)

    The code is based one the original regularized evolution open-source
    implementation:
    https://colab.research.google.com/github/google-research/google-research/blob/master/evolution/regularized_evolution_algorithm/regularized_evolution.ipynb

    Additional arguments on top of parent class
    :class:`~syne_tune.optimizer.schedulers.searchers.StochasticSearcher`:

    :param population_size: Size of the population, defaults to 100
    :param sample_size: Size of the candidate set to obtain a parent for the
        mutation, defaults to 10
    """

    def __init__(
        self,
        config_space,
        points_to_evaluate: Optional[List[dict]] = None,
        population_size: int = 100,
        sample_size: int = 10,
        random_seed: int = None,
    ):
        super(RegularizedEvolution, self).__init__(
            config_space=config_space,
            random_seed=random_seed,
            points_to_evaluate=points_to_evaluate,
        )
        assert (
            config_space_size(config_space) != 1
        ), f"config_space = {config_space} has size 1, does not offer any diversity"
        self.population_size = population_size
        self.sample_size = sample_size
        self.population = deque()
        self.random_state = np.random.RandomState(self.random_seed)

    def suggest(self, **kwargs) -> Optional[dict]:
        initial_config = self._next_points_to_evaluate()
        if initial_config is not None:
            return initial_config

        if len(self.population) < self.population_size:
            config = sample_random_config(
                config_space=self.config_space, rng=self.random_state
            )
        else:
            candidates = self.random_state.choice(
                list(self.population), size=self.sample_size
            )
            parent = min(candidates, key=lambda i: i.score)

            config = mutate_config(
                parent.config, config_space=self.config_space, rng=self.random_state
            )

        return config

    def on_trial_complete(
        self,
        trial_id: int,
        config: Dict[str, Any],
        metric: float,
    ):
        # Add element to the population
        element = PopulationElement(score=metric, config=config)
        self.population.append(element)

        # Remove the oldest element of the population.
        if len(self.population) > self.population_size:
            self.population.popleft()
