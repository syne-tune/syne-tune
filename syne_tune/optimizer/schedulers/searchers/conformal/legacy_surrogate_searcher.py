import logging
from collections import defaultdict
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

from syne_tune.config_space import Domain
from syne_tune.optimizer.scheduler import TrialSuggestion
from syne_tune.optimizer.schedulers.searchers import (
    impute_points_to_evaluate,
    StochasticSearcher,
)
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.quantile_regression_surrogate import (
    QuantileRegressionSurrogateModel,
)
from syne_tune.optimizer.schedulers.searchers.utils import make_hyperparameter_ranges
from syne_tune.util import catchtime

logger = logging.getLogger(__name__)


class LegacySurrogateSearcher(StochasticSearcher):
    def __init__(
        self,
        config_space: Dict,
        metric: str,
        mode: str = "min",
        num_init_random_draws: int = 5,
        update_frequency: int = 1,
        points_to_evaluate: Optional[List[Dict]] = None,
        max_fit_samples: int = None,
        random_seed: Optional[int] = None,
        **surrogate_kwargs,
    ):
        """
        Wrapper to use a single-fidelity surrogate as a multi-fidelity method by taking the last observation of each
        trial.
        :param num_init_random_draws: sampled at random until the number of observation exceeds this parameter.
        :param update_frequency: surrogates are only updated every `update_frequency` results, can be used to save
        scheduling time.
        :param points_to_evaluate: list of configuration to evaluate first.
        :param surrogate: surrogate to use, can be "GP", "QuantileRegression", "ConformalQuantileRegression", "BORE" or "REA".
        :param max_fit_samples: if the number of observation exceed this parameter, then `max_fit_samples` random samples
        are used to fit the model.
        :param random_seed:
        :param surrogate_kwargs:
        """
        super(LegacySurrogateSearcher, self).__init__(
            config_space=config_space,
            metric=metric,
            metric_names=[metric],
            mode=mode,
            points_to_evaluate=[],
            random_seed=random_seed,
        )
        assert mode in ["min", "max"]
        self.surrogate_kwargs = surrogate_kwargs
        self.num_init_random_draws = num_init_random_draws
        self.update_frequency = update_frequency
        self.mode = mode
        self.metric = metric
        self.trial_results = defaultdict(list)  # list of results for each trials
        self.trial_configs = {}
        self.hp_ranges = make_hyperparameter_ranges(config_space=config_space)
        self._points_to_evaluate = impute_points_to_evaluate(
            points_to_evaluate, config_space
        )
        self.surrogate_model = None
        self.index_last_result_fit = None
        self.new_candidates_sampled = False
        self.sampler = None
        self.max_fit_samples = max_fit_samples

    def get_config(self, trial_id: str, **kwargs) -> Optional[TrialSuggestion]:
        trial_id = int(trial_id)
        logger.debug(f"get_config trial {trial_id}, {self.num_results()} results")
        if self._points_to_evaluate:
            logger.debug(f"trial {trial_id}: pick from points to evaluate")
            config = self._points_to_evaluate.pop(0)
        else:
            if self.should_update():
                logger.debug(f"trial {trial_id}: fit model")
                with catchtime(f"fit model with {self.num_results()} observations"):
                    self.fit_model()
                self.index_last_result_fit = self.num_results()
            if self.surrogate_model is not None:
                logger.debug(f"trial {trial_id}: sample from model")
                config = self.surrogate_model.suggest()
            else:
                logger.debug(f"trial {trial_id}: sample at random")
                config = self.sample_random()
        self.trial_configs[trial_id] = config
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
        self.surrogate_model = QuantileRegressionSurrogateModel(
            config_space=self.config_space,
            max_fit_samples=self.max_fit_samples,
            random_state=self.random_state,
            mode=self.mode,
            min_samples_to_conformalize=32,
            valid_fraction=0.1,
            **self.surrogate_kwargs,
        )
        self.surrogate_model.fit(df_features=X, y=z)

    def on_trial_result(
        self, trial_id: str, config: Dict, result: Dict, update: bool = True
    ):
        trial_id = int(trial_id)
        y = result[self.metric]
        self.trial_results[trial_id].append(y)

    def sample_random(self) -> Dict:
        return {
            k: v.sample(random_state=self.random_state) if isinstance(v, Domain) else v
            for k, v in self.config_space.items()
        }

    def configs_to_df(self, configs: List[Dict]) -> pd.DataFrame:
        return pd.DataFrame(configs)

    def metric_names(self) -> List[str]:
        return [self.metric]

    def metric_mode(self) -> str:
        return self.mode

    def clone_from_state(self, state: dict):
        pass
