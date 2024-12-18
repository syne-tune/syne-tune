import numpy as np

from typing import Optional, List, Dict, Any
from collections import deque

from syne_tune.optimizer.schedulers.searchers.regularized_evolution import (
    PopulationElement,
    mutate_config,
    sample_random_config,
)
from syne_tune.config_space import config_space_size
from syne_tune.optimizer.schedulers.multiobjective.multiobjective_priority import (
    MOPriority,
    NonDominatedPriority,
)
from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher


class MultiObjectiveRegularizedEvolution(BaseSearcher):
    """
    Adapts regularized evolution algorithm by Real et al. to the multi-objective setting. Elements in the
    populations are scored via a multi-objective priority that is set to non-dominated sort by default. Parents are sampled from the population based on
    this score.

    Additional arguments on top of parent class
    :class:`syne_tune.optimizer.schedulers.searchers.StochasticSearcher`:

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
        multiobjective_priority: Optional[MOPriority] = None,
        random_seed: int = None,
    ):

        super(MultiObjectiveRegularizedEvolution, self).__init__(
            config_space=config_space,
            points_to_evaluate=points_to_evaluate,
            random_seed=random_seed,
        )
        assert (
            config_space_size(config_space) != 1
        ), f"config_space = {config_space} has size 1, does not offer any diversity"
        self.population_size = population_size
        self.sample_size = sample_size
        self.population = deque()

        self.random_state = np.random.RandomState(self.random_seed)

        if multiobjective_priority is None:
            self._multiobjective_priority = NonDominatedPriority()
        else:
            self._multiobjective_priority = multiobjective_priority

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
        metrics: List[float],
    ):

        # Add element to the population
        element = PopulationElement(results=metrics, config=config)
        self.population.append(element)

        metric_recorded = np.empty((len(self.population), len(metrics)))
        for i, pi in enumerate(self.population):
            y = np.array(pi.results)
            metric_recorded[i, :] = y

        priorities = self._multiobjective_priority(metric_recorded)
        for i, pi in enumerate(self.population):
            pi.score = priorities[i]

        # Remove the oldest element of the population.
        if len(self.population) > self.population_size:
            self.population.popleft()
