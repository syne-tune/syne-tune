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
from typing import Dict, Optional, List
import logging

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.models.transforms import Warp
from botorch.utils import standardize
from botorch.utils.transforms import normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.utils.errors import NotPSDError

import syne_tune.config_space as cs
from syne_tune.optimizer.schedulers.searchers import SearcherWithRandomSeed

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory import make_hyperparameter_ranges


logger = logging.getLogger(__name__)


NOISE_LEVEL = 1e-3

class BotorchSearcher(SearcherWithRandomSeed):

    def __init__(
            self,
            config_space: Dict,
            metric: str,
            num_init_random_draws: int = 3,
            mode: str = "min",
            points_to_evaluate: Optional[List[Dict]] = None,
            fantasising: bool = True,
            max_num_observations: Optional[int] = 200,
            input_warping: bool = True,
            **kwargs,
    ):
        """
        A searcher that suggest configurations using BOTORCH to build GP surrogate and optimize acquisition function.
        `qExpectedImprovement is used for the acquisition function given that it supports pending evaluations.
        :param config_space: configuration space to optimize
        :param metric: metric to optimize, should be present in reported results.
        :param num_init_random_draws: number of initial random draws, after this number the suggestion are obtained
        from the GP surrogate model.
        :param mode: 'min' or 'max'
        :param points_to_evaluate: if passed, those configurations are evaluated first
        :param fantasising: whether to fantasize pending evaluations by sampling from the GP posterior
        :param max_num_observations: maximum number of observation to use when fitting the GP, if the number of
        observations gets larger than this number, then data is subsampled. If None, then all data is used to fit the GP.
        :param input_warping: whether to apply input warping when fitting the GP.
        :param kwargs: additional arguments of SearcherWithRandomSeed
        """
        super(BotorchSearcher, self).__init__(config_space, metric, points_to_evaluate=points_to_evaluate, **kwargs)
        assert num_init_random_draws >= 2
        assert mode in ['min', 'max']
        self.hp_ranges = make_hyperparameter_ranges(config_space=config_space)
        self.mode = mode
        self.metric_name = metric
        self.num_minimum_observations = num_init_random_draws
        self.points_to_evaluate = points_to_evaluate
        self.config_seen = set()
        self.fantasising = fantasising
        self.max_num_observations = max_num_observations
        self.input_warping = input_warping
        self.trial_configs = {}
        self.pending_trials = set()
        self.trial_observations = {}

    def _update(self, trial_id: str, config: Dict, result: Dict):
        trial_id = int(trial_id)
        self.trial_observations[trial_id] = result[self.metric_name]
        self.pending_trials.remove(trial_id)

    def clone_from_state(self, state):
        pass

    def num_suggestions(self):
        return len(self.trial_configs)

    def get_config(self, trial_id: str, **kwargs):
        trial_id = int(trial_id)
        config_suggested = self._next_initial_config()

        if config_suggested is None:
            if self.points_to_evaluate is not None and self.num_suggestions() < len(self.points_to_evaluate):
                # if we are not done yet with points_to_evaluate, we pick the next one from this list
                config_suggested = self.points_to_evaluate[self.num_suggestions()]
            else:
                not_enough_suggestion = len(self.objectives()) < self.num_minimum_observations
                if not_enough_suggestion:
                    config_suggested = self._sample_random()
                else:
                    config_suggested = self._sample_next_candidate()

        self.pending_trials.add(trial_id)
        self.trial_configs[trial_id] = config_suggested
        self.config_seen.add(tuple(config_suggested.values()))

        return config_suggested

    def _sample_next_candidate(self) -> Dict:
        """
        :return: a next candidate to evaluate, if possible it is obtained by fitting a GP on past data and maximizing EI
        if this fails because of numerical difficulties with non PSD matrices, then the candidate is sampled at random.
        """
        try:
            X = np.array(self._config_to_feature_matrix(self._configs_with_results()))
            y = self.objectives()

            if self.max_num_observations is not None and len(X) >= self.max_num_observations:
                perm = self.random_state.permutation(len(X))[:self.max_num_observations]
                X = X[perm]
                y = y[perm]
                subsample = True
            else:
                subsample = False

            X_tensor = torch.Tensor(X)
            Y_tensor = standardize(torch.Tensor(y).reshape(-1, 1))
            gp = self._make_gp(X_tensor=X_tensor, Y_tensor=Y_tensor)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll, max_retries=0)

            if self.pending_trials and self.fantasising and not subsample:
                X_pending = self._config_to_feature_matrix(self._configs_pending())
            else:
                X_pending = None

            acq = qExpectedImprovement(
                model=gp,
                best_f=Y_tensor.min().item() if self.mode == 'min' else Y_tensor.max().item(),
                maximize=self.mode == 'max',
                X_pending=X_pending,
            )

            candidate, acq_value = optimize_acqf(
                acq,
                bounds=torch.Tensor(self.hp_ranges.get_ndarray_bounds()).T,
                q=1,
                num_restarts=3,
                raw_samples=100
            )

            candidate = candidate.detach().numpy()[0]
            config = self.hp_ranges.from_ndarray(candidate)
            if not self._is_config_already_seen(config):
                return config
            else:
                logger.warning("Optimization of the acquisition function yielded a config that was already seen.")
                return self._sample_and_pick_acq_best(acq)
        except NotPSDError as _:
            logging.warning("Chlolesky inversion failed, sampling randomly.")
            return self._sample_random()

    def _make_gp(self, X_tensor: torch.Tensor, Y_tensor: torch.Tensor) -> SingleTaskGP:
        double_precision = False
        if double_precision:
            X_tensor = X_tensor.double()
            Y_tensor = Y_tensor.double()

        noise_std = NOISE_LEVEL
        Y_tensor += noise_std * torch.randn_like(Y_tensor)

        if self.input_warping:
            warp_tf = Warp(indices=list(range(X_tensor.shape[-1])))
        else:
            warp_tf = None
        return SingleTaskGP(X_tensor, Y_tensor, input_transform=warp_tf)

    def _config_to_feature_matrix(self, configs: List[Dict]) -> torch.Tensor:
        bounds = torch.Tensor(self.hp_ranges.get_ndarray_bounds()).T
        X = torch.Tensor([self.hp_ranges.to_ndarray(config) for config in configs])
        return normalize(X, bounds)

    def objectives(self):
        return np.array(list(self.trial_observations.values()))

    def _sample_and_pick_acq_best(self, acq, num_samples: int = 100) -> Dict:
        """
        :param acq:
        :param num_samples:
        :return: Samples `num_samples` candidates and return the one maximizing the acquisitition function `acq` that
        was not seen earlier, if all samples were seen, return a random sample instead.
        """
        configs_candidates = [self._sample_random() for _ in range(num_samples)]
        configs_candidates = [x for x in configs_candidates if not self._is_config_already_seen(x)]
        logger.debug(f"Sampling among {len(configs_candidates)} unseen configs")
        if configs_candidates:
            X_tensor = self._config_to_feature_matrix(configs_candidates)
            ei = acq(X_tensor.unsqueeze(dim=-2))
            return configs_candidates[ei.argmax()]
        else:
            return self._sample_random()

    def _is_config_already_seen(self, config) -> bool:
        return tuple(config.values()) in self.config_seen

    def _sample_random(self) -> Dict:
        return {
            k: v.sample(random_state=self.random_state)
            if isinstance(v, cs.Domain) else v
            for k, v in self.config_space.items()
        }

    def _configs_with_results(self) -> List[Dict]:
        return [
            config
            for trial, config in self.trial_configs.items()
            if not trial in self.pending_trials
        ]

    def _configs_pending(self) -> List[Dict]:
        return [
            config
            for trial, config in self.trial_configs.items()
            if trial in self.pending_trials
        ]

    def metric_names(self) -> List[str]:
        return [self.metric_name]

    def metric_mode(self) -> str:
        return self.mode