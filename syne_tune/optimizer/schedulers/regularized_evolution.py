import numpy as np
import copy

from collections import deque
from typing import Optional, Dict, List
from dataclasses import dataclass

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import TrialScheduler, TrialSuggestion
from syne_tune.search_space import Domain, Categorical, Float, Integer


@dataclass
class PopulationElement:
    score: int = 0
    config: dict = None


class RegularizedEvolution(TrialScheduler):
    def __init__(
            self,
            config_space,
            metric: str,
            mode: str,
            population_size: int = 100,
            sample_size: int = 10,
            random_seed: int = None
    ):
        """
        Implements the regularized evolution algorithm proposed by Real et al. The original implementation only
        considers categorical hyperparameters. For integer and float parameters we sample a new value uniformly
        at random for the entire search space.

        Real, E., Aggarwal, A., Huang, Y., and Le, Q. V.
        Regularized Evolution for Image Classifier Architecture Search.
        In Proceedings of the Conference on Artificial Intelligence (AAAI’19)

        The code is based one the original regularized evolution open-source implementation:
        https://colab.research.google.com/github/google-research/google-research/blob/master/evolution/regularized_evolution_algorithm/regularized_evolution.ipynb


        Parameters
        ----------
        config_space: dict
            Configuration space for trial evaluation function
        metric : str
            Name of metric to optimize, key in result's obtained via
            `on_trial_result`
        mode : str
            Mode to use for the metric given, can be 'min' or 'max', default to 'min'.
        population_size : int
            Size of the population.
        sample_size : int
            Size of the candidate set to obtain a parent for the mutation.
        random_seed : int
            Seed for the random number generation. If set to None, use random seed.
        """
        super(RegularizedEvolution, self).__init__(config_space=config_space)
        self.metric = metric
        self.mode = mode
        self.population_size = population_size
        self.sample_size = sample_size
        self.population = deque()
        if random_seed is None:
            random_seed = np.random.randint(0, 2 ** 32)
        self._random_state = np.random.RandomState(random_seed)

    def mutate_arch(self, config):

        child_config = copy.deepcopy(config)

        # pick random hyperparameter and mutate it
        hypers = []
        for k, v in self.config_space.items():
            if isinstance(v, Domain):
                hypers.append(k)
        name = self._random_state.choice(hypers)
        hyperparameter = self.config_space[name]

        if isinstance(hyperparameter, Categorical):
            # drop current values from potential choices to not sample the same value again
            choices = copy.deepcopy(hyperparameter.categories)
            choices.remove(config[name])

            new_value = self._random_state.choice(choices)

        elif isinstance(hyperparameter, Float):
            new_value = Float.sample(random_state=self._random_state)

        elif isinstance(hyperparameter, Integer):
            new_value = Integer.sample(random_state=self._random_state)

        child_config[name] = new_value

        return child_config

    def sample_random_config(self):
        return {k: v.sample(random_state=self._random_state) if isinstance(v, Domain) else v for k, v in
                self.config_space.items()}

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:

        if len(self.population) < self.population_size:
            config = self.sample_random_config()
        else:
            candidates = self._random_state.choice(list(self.population), size=self.sample_size)
            parent = max(candidates, key=lambda i: i.score)

            config = self.mutate_arch(parent.config)

        return TrialSuggestion.start_suggestion(config=config)

    def on_trial_complete(self, trial: Trial, result: Dict):

        score = result[self.metric]

        if self.mode == 'min':
            score *= -1

        # Add element to the population
        element = PopulationElement(score=score, config=trial.config)
        self.population.append(element)

        # Remove the oldest element of the population.
        if len(self.population) > self.population_size:
            self.population.popleft()

    def metric_names(self) -> List[str]:
        return [self.metric]

    def metric_mode(self) -> str:
        return self.mode
