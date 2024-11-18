import numpy as np

from typing import Optional, List, Union

from syne_tune.optimizer.schedulers.searchers.legacy_regularized_evolution import (
    PopulationElement,
    LegacyRegularizedEvolution,
)

from syne_tune.optimizer.schedulers.multiobjective.multiobjective_priority import (
    MOPriority,
    NonDominatedPriority,
)


class LegacyMultiObjectiveRegularizedEvolution(LegacyRegularizedEvolution):
    """
    Adapts regularized evolution algorithm by Real et al. to the multi-objective setting. Elements in the
    populations are scored via a multi-objective priority that is set to non-dominated sort by default. Parents are sampled from the population based on
    this score.

    Additional arguments on top of parent class
    :class:`syne_tune.optimizer.schedulers.searchers.StochasticSearcher`:

    :param mode: Mode to use for the metric given, can be "min" or "max",
        defaults to "min"
    :param population_size: Size of the population, defaults to 100
    :param sample_size: Size of the candidate set to obtain a parent for the
        mutation, defaults to 10
    """

    def __init__(
        self,
        config_space,
        metric: List[str],
        mode: Union[List[str], str],
        points_to_evaluate: Optional[List[dict]] = None,
        population_size: int = 100,
        sample_size: int = 10,
        multiobjective_priority: Optional[MOPriority] = None,
        **kwargs,
    ):

        if isinstance(mode, str):
            mode = [mode] * len(metric)

        super(LegacyMultiObjectiveRegularizedEvolution, self).__init__(
            config_space,
            metric=metric,
            points_to_evaluate=points_to_evaluate,
            population_size=population_size,
            sample_size=sample_size,
            mode=mode,
            **kwargs,
        )

        if multiobjective_priority is None:
            self._multiobjective_priority = NonDominatedPriority()
        else:
            self._multiobjective_priority = multiobjective_priority

    def _update(self, trial_id: str, config: dict, result: dict):
        results = {}
        for mode, metric in zip(self._mode, self._metric):
            value = result[metric]
            if mode == "max":
                value *= -1
            results[metric] = value

        # Add element to the population
        element = PopulationElement(result=results, config=config)
        self.population.append(element)

        metric_recorded = np.empty((len(self.population), len(self._metric)))
        for i, pi in enumerate(self.population):
            y = np.array(list(pi.result.values()))
            metric_recorded[i, :] = y

        priorities = self._multiobjective_priority(metric_recorded)
        for i, pi in enumerate(self.population):
            pi.score = priorities[i]

        # Remove the oldest element of the population.
        if len(self.population) > self.population_size:
            self.population.popleft()
