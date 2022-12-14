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
import copy
import logging

from collections import deque
from typing import Optional, List
from dataclasses import dataclass

from syne_tune.optimizer.schedulers.searchers import SearcherWithRandomSeed
from syne_tune.config_space import Domain


@dataclass
class PopulationElement:
    score: int = 0
    config: dict = None


class RegularizedEvolution(SearcherWithRandomSeed):
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
    :class:`~syne_tune.optimizer.schedulers.searchers.SearcherWithRandomSeed`:

    :param mode: Mode to use for the metric given, can be "min" or "max",
        defaults to "min"
    :param population_size: Size of the population, defaults to 100
    :param sample_size: Size of the candidate set to obtain a parent for the
        mutation, defaults to 10
    """

    def __init__(
        self,
        config_space,
        metric: str,
        points_to_evaluate: Optional[List[dict]] = None,
        mode: str = "min",
        population_size: int = 100,
        sample_size: int = 10,
        **kwargs,
    ):
        super(RegularizedEvolution, self).__init__(
            config_space, metric, points_to_evaluate=points_to_evaluate, **kwargs
        )
        self.mode = mode
        self.population_size = population_size
        self.sample_size = sample_size
        self.population = deque()
        self.num_sample_try = 1000  # number of times allowed to sample a mutation

    def _mutate_config(self, config: dict) -> dict:
        child_config = copy.deepcopy(config)

        # sample mutation until a different configuration is found
        for sample_try in range(self.num_sample_try):
            if child_config == config:
                # sample a random hyperparameter to mutate
                hps = [
                    (k, v)
                    for k, v in self.config_space.items()
                    if isinstance(v, Domain) and len(v) > 1
                ]
                assert (
                    len(hps) >= 0
                ), "all hyperparameters only have a single value, cannot perform mutations."
                hp_name, hp = hps[self.random_state.randint(len(hps))]

                # mutate the value by sampling
                config[hp_name] = hp.sample(random_state=self.random_state)
            else:
                break
        if sample_try == self.num_sample_try:
            logging.info(
                f"Did not manage to sample a different configuration with {self.num_sample_try}, "
                f"sampling at random"
            )
            return self._sample_random_config()

        return child_config

    def _sample_random_config(self) -> dict:
        return {
            k: v.sample(random_state=self.random_state) if isinstance(v, Domain) else v
            for k, v in self.config_space.items()
        }

    def get_config(self, **kwargs) -> Optional[dict]:
        initial_config = self._next_initial_config()
        if initial_config is not None:
            return initial_config

        if len(self.population) < self.population_size:
            config = self._sample_random_config()
        else:
            candidates = self.random_state.choice(
                list(self.population), size=self.sample_size
            )
            parent = min(candidates, key=lambda i: i.score)

            config = self._mutate_config(parent.config)

        return config

    def _update(self, trial_id: str, config: dict, result: dict):
        score = result[self._metric]

        if self.mode == "max":
            score *= -1

        # Add element to the population
        element = PopulationElement(score=score, config=config)
        self.population.append(element)

        # Remove the oldest element of the population.
        if len(self.population) > self.population_size:
            self.population.popleft()

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers.fifo import FIFOScheduler

        assert isinstance(
            scheduler, FIFOScheduler
        ), "This searcher requires FIFOScheduler scheduler"
        super().configure_scheduler(scheduler)

    def clone_from_state(self, state: dict):
        raise NotImplementedError
