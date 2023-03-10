# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import numpy as np

from typing import Optional, List, Union

from syne_tune.optimizer.schedulers.searchers.regularized_evolution import (
    PopulationElement,
    RegularizedEvolution,
)

from syne_tune.optimizer.schedulers.multiobjective.multiobjective_priority import (
    MOPriority,
    NonDominatedPriority,
)


class MultiObjectiveRegularizedEvolution(RegularizedEvolution):
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
            modes = [mode] * len(metric)
        else:
            modes = mode

        super(MultiObjectiveRegularizedEvolution, self).__init__(
            config_space,
            metric=metric,
            mode=modes,
            points_to_evaluate=points_to_evaluate,
            population_size=population_size,
            sample_size=sample_size,
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
