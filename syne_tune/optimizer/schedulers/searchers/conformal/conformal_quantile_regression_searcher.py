import logging
from collections import defaultdict
from typing import Dict, Optional, List, Any

import numpy as np
import pandas as pd

from syne_tune.config_space import Domain

from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.surrogate_model import (
    SurrogateModel,
)
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.quantile_regression_surrogate import (
    QuantileRegressionSurrogateModel,
)
from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import (
    SingleObjectiveBaseSearcher,
)
from syne_tune.optimizer.schedulers.searchers.utils import make_hyperparameter_ranges
from syne_tune.util import catchtime

logger = logging.getLogger(__name__)


class ConformalQuantileRegression(SingleObjectiveBaseSearcher):
    def __init__(
        self,
        config_space: Dict,
        random_seed: Optional[int] = None,
        points_to_evaluate: Optional[List[Dict]] = None,
        num_init_random_draws: int = 5,
        update_frequency: int = 1,
        max_fit_samples: int = None,
        surrogate_cls: SurrogateModel = QuantileRegressionSurrogateModel,
        **surrogate_kwargs,
    ):
        """
        Bayesian optimization using conformalized quantile regression models as proposed by:

            Optimizing hyperparameters with conformal quantile regression
            D. Salinas, J. Golebiowski, A. Klein, M. Seeger, C. Archambeau
            International Conference on Machine Learning 2023

        :param config_space: Configuration space for the evaluation function.
        :param random_seed: Seed for initializing random number generators.
        :param points_to_evaluate: A set of initial configurations to be evaluated before starting the optimization.
        :param num_init_random_draws: sampled at random until the number of observation exceeds this parameter.
        :param update_frequency: surrogates are only updated every `update_frequency` results, can be used to save
        scheduling time.
        :param max_fit_samples: if the number of observation exceed this parameter, then `max_fit_samples` random samples
        are used to fit the model.
        :param surrogate_cls: SurrogateModel class to model the objective function
        :param surrogate_kwargs: additional kwargs for the surrogate model
        """
        super(ConformalQuantileRegression, self).__init__(
            config_space=config_space,
            points_to_evaluate=points_to_evaluate,
            random_seed=random_seed,
        )
        self.surrogate_kwargs = surrogate_kwargs
        self.num_init_random_draws = num_init_random_draws
        self.update_frequency = update_frequency
        self.trial_results = defaultdict(list)  # list of results for each trials
        self.trial_configs = {}
        self.hp_ranges = make_hyperparameter_ranges(config_space=config_space)
        self.surrogate_model = None
        self.index_last_result_fit = None
        self.new_candidates_sampled = False
        self.sampler = None
        self.max_fit_samples = max_fit_samples
        self.surrogate_cls = surrogate_cls
        self.random_state = np.random.RandomState(self.random_seed)

    def suggest(self, **kwargs) -> Optional[Dict[str, Any]]:
        config = self._next_points_to_evaluate()

        if config is None:
            if self.should_update():
                logger.debug(f"fit model")
                with catchtime(f"fit model with {self.num_results()} observations"):
                    self.fit_model()
                self.index_last_result_fit = self.num_results()
            if self.surrogate_model is not None:
                logger.debug(f"sample from model")
                config = self.surrogate_model.suggest()
            else:
                logger.debug(f"sample at random")
                config = self.sample_random()
        return config

    def should_update(self) -> bool:
        enough_observations = self.num_results() >= self.num_init_random_draws
        if enough_observations:
            if self.index_last_result_fit is None:
                return True
            else:
                new_results_seen_since_last_fit = (
                    self.num_results() - self.index_last_result_fit
                )
                return new_results_seen_since_last_fit >= self.update_frequency
        else:
            return False

    def num_results(self) -> int:
        return len(self.trial_results)

    def make_input_target(self):
        configs = [
            self.trial_configs[trial_id] for trial_id in self.trial_results.keys()
        ]
        X = self.configs_to_df(configs)
        # takes the last value of each fidelity for each trial
        z = np.array([trial_values[-1] for trial_values in self.trial_results.values()])
        return X, z

    def fit_model(self):
        X, z = self.make_input_target()
        self.surrogate_model = self.surrogate_cls(
            config_space=self.config_space,
            max_fit_samples=self.max_fit_samples,
            random_state=self.random_state,
            mode="min",
            min_samples_to_conformalize=32,
            valid_fraction=0.1,
            **self.surrogate_kwargs,
        )
        self.surrogate_model.fit(df_features=X, y=z)

    def on_trial_complete(
        self,
        trial_id: int,
        config: Dict[str, Any],
        metric: float,
        resource_level: int = None,
    ):
        self.trial_configs[trial_id] = config
        self.trial_results[trial_id].append(metric)

    def sample_random(self) -> Dict:
        return {
            k: v.sample(random_state=self.random_state) if isinstance(v, Domain) else v
            for k, v in self.config_space.items()
        }

    def configs_to_df(self, configs: List[Dict]) -> pd.DataFrame:
        return pd.DataFrame(configs)
