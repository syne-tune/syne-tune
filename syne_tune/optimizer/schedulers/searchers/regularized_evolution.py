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
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

from syne_tune.optimizer.schedulers.searchers import StochasticSearcher
from syne_tune.config_space import config_space_size, Domain
from syne_tune.optimizer.schedulers.searchers.utils import make_hyperparameter_ranges
from syne_tune.optimizer.schedulers.searchers.searcher_base import (
    sample_random_configuration,
)

logger = logging.getLogger(__name__)


@dataclass
class PopulationElement:
    result: Dict[str, Any] = None
    score: int = 0
    config: Dict[str, Any] = None


class RegularizedEvolution(StochasticSearcher):
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

    :param mode: Mode to use for the metric given, can be "min" or "max",
        defaults to "min"
    :param population_size: Size of the population, defaults to 100
    :param sample_size: Size of the candidate set to obtain a parent for the
        mutation, defaults to 10
    """

    def __init__(
        self,
        config_space,
        metric: Union[List[str], str],
        points_to_evaluate: Optional[List[dict]] = None,
        population_size: int = 100,
        sample_size: int = 10,
        **kwargs,
    ):
        super(RegularizedEvolution, self).__init__(
            config_space, metric, points_to_evaluate=points_to_evaluate, **kwargs
        )
        assert (
            config_space_size(config_space) != 1
        ), f"config_space = {config_space} has size 1, does not offer any diversity"
        self.population_size = population_size
        self.sample_size = sample_size
        self.population = deque()
        self.num_sample_try = 1000  # number of times allowed to sample a mutation
        self._hp_ranges = make_hyperparameter_ranges(self.config_space)
        allow_duplicates = kwargs.get("allow_duplicates")
        if allow_duplicates is not None and (not allow_duplicates):
            logger.warning(
                "This class does not support allow_duplicates argument. Sampling is with replacement"
            )
        if kwargs.get("restrict_configurations") is not None:
            logger.warning("This class does not support restrict_configurations")
        self._non_constant_hps = [
            name
            for name, domain in config_space.items()
            if isinstance(domain, Domain) and len(domain) != 1
        ]

    def _mutate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        child_config = copy.deepcopy(config)
        config_ms = self._hp_ranges.config_to_match_string(config)
        # Sample mutation until a different configuration is found
        config_is_mutated = False
        for _ in range(self.num_sample_try):
            if self._hp_ranges.config_to_match_string(child_config) == config_ms:
                # Sample a random hyperparameter to mutate
                hp_name = self.random_state.choice(self._non_constant_hps, 1).item()
                # Mutate the value by sampling
                child_config[hp_name] = self.config_space[hp_name].sample(
                    random_state=self.random_state
                )
            else:
                config_is_mutated = True
                break
        if not config_is_mutated:
            logger.info(
                "Did not manage to sample a different configuration with "
                f"{self.num_sample_try}, sampling at random"
            )
            child_config = self._sample_random_config()

        return child_config

    def _sample_random_config(self) -> Dict[str, Any]:
        return sample_random_configuration(self._hp_ranges, self.random_state)

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

    def _update(self, trial_id: str, config: Dict[str, Any], result: Dict[str, Any]):
        score = result[self._metric]

        if self._mode == "max":
            score *= -1

        # Add element to the population
        element = PopulationElement(result=result, score=score, config=config)
        self.population.append(element)

        # Remove the oldest element of the population.
        if len(self.population) > self.population_size:
            self.population.popleft()

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers.scheduler_searcher import (
            TrialSchedulerWithSearcher,
        )

        assert isinstance(
            scheduler, TrialSchedulerWithSearcher
        ), "This searcher requires TrialSchedulerWithSearcher scheduler"
        super().configure_scheduler(scheduler)

    def clone_from_state(self, state: Dict[str, Any]):
        raise NotImplementedError
