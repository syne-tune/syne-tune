import logging

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List, Union

from syne_tune.config_space import Domain


class SurrogateModel:
    def __init__(
        self,
        config_space: Dict,
        mode: str,
        max_fit_samples: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None,
    ):
        self.random_state = random_state if random_state else np.random
        self.config_space = config_space
        self.mode = mode
        self.config_candidates = []
        self.config_seen = set()
        self._sampler = None
        self.max_fit_samples = max_fit_samples

    def suggest(self, replace_config: bool = False) -> dict:
        if not self._sampler:
            return self._sample_random()

        config_idx = self._sample_best()
        config = self.config_candidates[config_idx]

        if replace_config:
            self._replace_config_by_new_sample(config_idx)
            self.config_seen.add(tuple(config.values()))
        return config

    def fit(
        self,
        df_features: pd.DataFrame,
        y: np.array,
        ncandidates: Union[int, pd.DataFrame] = 2000,
    ):
        self._fit(df_features=df_features, y=y)

        if isinstance(ncandidates, int):
            self._update_candidates(n_candidates=ncandidates)
        elif isinstance(ncandidates, pd.DataFrame):
            self.df_candidates = ncandidates
            self.config_candidates = ncandidates.to_dict(orient="records")
        else:
            raise ValueError(f"wrong type for {ncandidates}")
        self._sampler = self._get_sampler(self.df_candidates)

    def _fit(self, df_features: pd.DataFrame, y: np.array):
        """
        :param df_features: input features with shape (n, d)
        :param y: expected output with shape (n, 1)
        TODO unify
        :return:
        """
        pass

    def _sample_best(self) -> int:
        residual_samples = self._surrogate_pred()
        if self.mode == "max":
            residual_samples *= -1
        return np.argmin(residual_samples)

    def _surrogate_pred(self):
        z_pred = self._sampler()
        return z_pred

    def predict(self, df_features: pd.DataFrame) -> Tuple[np.array, np.array]:
        """
        :param df_features: input features to make predictions with shape (n, d)
        :return: predictions in the shape of (n,)
        # TODO should we rather have (n, 1)? input is taken in this form
        """
        # get mean/std predicted of y | x
        pass

    def _get_sampler(self, df_features: pd.DataFrame) -> np.array:
        # avoid computing mu, sigma between multiple calls
        mu, std = self.predict(df_features)

        def sample():
            random_state = (
                self.random_state if self.random_state is not None else np.random
            )
            return random_state.normal(mu, std)

        return sample

    def _update_candidates(self, n_candidates: int = 2000) -> None:
        # print("update candidates")
        self.config_candidates = [self._sample_random() for _ in range(n_candidates)]
        self.df_candidates = self._configs_to_df(self.config_candidates)

    def _sample_random_unseen(self, num_tries: int = 100):
        for i in range(num_tries):
            new_config = self._sample_random()
            if not self._config_already_seen(new_config):
                break
        if i == num_tries:
            logging.warning(f"could not sample an unseen config in {num_tries} tries.")
        return new_config

    def _replace_config_by_new_sample(self, config_idx: int):
        assert config_idx < len(self.config_candidates)
        # replace the config selected by a new one
        self.config_candidates[config_idx] = self._sample_random_unseen()

        # Once the candidates are updated, we need to update df candidates and the sampler
        self.df_candidates = self._configs_to_df(self.config_candidates)
        self._sampler = self._get_sampler(self.df_candidates)

    def _sample_random(self) -> Dict:
        return {
            k: v.sample(random_state=self.random_state) if isinstance(v, Domain) else v
            for k, v in self.config_space.items()
        }

    def _configs_to_df(self, configs: List[Dict]) -> pd.DataFrame:
        return pd.DataFrame(configs)

    def _config_already_seen(self, config) -> bool:
        return tuple(config.values()) in self.config_seen
