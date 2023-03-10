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
from typing import Optional, List, Dict, Any
import logging

import numpy as np
from syne_tune.try_import import try_import_botorch_message

try:
    from torch import Tensor, randn_like, random
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.transforms import Warp
    from botorch.utils import standardize
    from botorch.utils.transforms import normalize
    from botorch.acquisition import qExpectedImprovement
    from botorch.optim import optimize_acqf
    from botorch.exceptions.errors import ModelFittingError
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from linear_operator.utils.errors import NotPSDError
except ImportError:
    print(try_import_botorch_message())

from syne_tune.optimizer.schedulers.searchers import (
    StochasticAndFilterDuplicatesSearcher,
)

logger = logging.getLogger(__name__)


NOISE_LEVEL = 1e-3


class BoTorchSearcher(StochasticAndFilterDuplicatesSearcher):
    """
    A searcher that suggest configurations using BOTORCH to build GP surrogate
    and optimize acquisition function.

    ``qExpectedImprovement`` is used for the acquisition function, given that it
    supports pending evaluations.

    Additional arguments on top of parent class
    :class:`~syne_tune.optimizer.schedulers.searchers.StochasticAndFilterDuplicatesSearcher`:

    :param mode: "min" (default) or "max"
    :param num_init_random: :meth:`get_config` returns randomly drawn
        configurations until at least ``init_random`` observations have been
        recorded in :meth:`update`. After that, the BOTorch algorithm is used.
        Defaults to 3
    :param no_fantasizing: If ``True``, fantasizing is not done and pending
        evaluations are ignored. This may lead to loss of diversity in
        decisions. Defaults to ``False``
    :param max_num_observations: Maximum number of observation to use when
        fitting the GP. If the number of observations gets larger than this
        number, then data is subsampled. If ``None``, then all data is used to
        fit the GP. Defaults to 200
    :param input_warping: Whether to apply input warping when fitting the GP.
        Defaults to ``True``
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        points_to_evaluate: Optional[List[dict]] = None,
        allow_duplicates: bool = False,
        restrict_configurations: Optional[List[Dict[str, Any]]] = None,
        mode: str = "min",
        num_init_random: int = 3,
        no_fantasizing: bool = False,
        max_num_observations: Optional[int] = 200,
        input_warping: bool = True,
        **kwargs,
    ):
        super(BoTorchSearcher, self).__init__(
            config_space,
            metric=metric,
            points_to_evaluate=points_to_evaluate,
            allow_duplicates=allow_duplicates,
            restrict_configurations=restrict_configurations,
            **kwargs,
        )
        assert num_init_random >= 2
        assert mode in ["min", "max"]
        self.num_minimum_observations = num_init_random
        self.fantasising = not no_fantasizing
        self.max_num_observations = max_num_observations
        self.input_warping = input_warping
        self.trial_configs = dict()
        self.pending_trials = set()
        self.trial_observations = dict()

        # Set the random seed for botorch as well
        if "random_seed" in kwargs:
            random.manual_seed(kwargs["random_seed"])

    def _update(self, trial_id: str, config: Dict[str, Any], result: Dict[str, Any]):
        trial_id = int(trial_id)
        self.trial_observations[trial_id] = result[self._metric]
        if trial_id in self.pending_trials:
            self.pending_trials.remove(trial_id)

    def clone_from_state(self, state: Dict[str, Any]):
        raise NotImplementedError

    def num_suggestions(self):
        return len(self.trial_configs)

    def _get_config(self, trial_id: str, **kwargs) -> Optional[dict]:
        trial_id = int(trial_id)
        config_suggested = self._next_initial_config()

        if config_suggested is None:
            if len(self.objectives()) < self.num_minimum_observations:
                config_suggested = self._get_random_config()
            else:
                config_suggested = self._sample_next_candidate()

        if config_suggested is not None:
            self.trial_configs[trial_id] = config_suggested

        return config_suggested

    def register_pending(
        self,
        trial_id: str,
        config: Optional[dict] = None,
        milestone: Optional[int] = None,
    ):
        super().register_pending(trial_id, config, milestone)
        self.pending_trials.add(int(trial_id))

    def evaluation_failed(self, trial_id: str):
        super().evaluation_failed(trial_id)
        self.cleanup_pending(trial_id)

    def cleanup_pending(self, trial_id: str):
        trial_id = int(trial_id)
        if trial_id in self.pending_trials:
            self.pending_trials.remove(trial_id)

    def dataset_size(self):
        return len(self.trial_observations)

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers.scheduler_searcher import (
            TrialSchedulerWithSearcher,
        )

        assert isinstance(
            scheduler, TrialSchedulerWithSearcher
        ), "This searcher requires TrialSchedulerWithSearcher scheduler"
        super().configure_scheduler(scheduler)

    def _sample_next_candidate(self) -> Optional[dict]:
        """
        :return: A next candidate to evaluate, if possible it is obtained by
            fitting a GP on past data and maximizing EI. If this fails because
            of numerical difficulties with non PSD matrices, then the candidate
            is sampled at random.
        """
        try:
            X = np.array(self._config_to_feature_matrix(self._configs_with_results()))
            y = self.objectives()
            if self._mode == "min":
                # qExpectedImprovement only supports maximization
                y *= -1

            if (
                self.max_num_observations is not None
                and len(X) >= self.max_num_observations
            ):
                perm = self.random_state.permutation(len(X))[
                    : self.max_num_observations
                ]
                X = X[perm]
                y = y[perm]
                subsample = True
            else:
                subsample = False

            X_tensor = Tensor(X)
            Y_tensor = standardize(Tensor(y).reshape(-1, 1))
            gp = self._make_gp(X_tensor=X_tensor, Y_tensor=Y_tensor)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll, max_attempts=1)

            if self.pending_trials and self.fantasising and not subsample:
                X_pending = self._config_to_feature_matrix(self._configs_pending())
            else:
                X_pending = None

            acq = qExpectedImprovement(
                model=gp,
                best_f=Y_tensor.max().item(),
                X_pending=X_pending,
            )

            config = None
            if self._restrict_configurations is None:
                # Continuous optimization of acquisition function only if
                # ``restrict_configurations`` not used
                candidate, acq_value = optimize_acqf(
                    acq,
                    bounds=Tensor(self._hp_ranges.get_ndarray_bounds()).T,
                    q=1,
                    num_restarts=3,
                    raw_samples=100,
                )
                candidate = candidate.detach().numpy()[0]
                config = self._hp_ranges.from_ndarray(candidate)
                if self.should_not_suggest(config):
                    logger.warning(
                        "Optimization of the acquisition function yielded a config that was already seen."
                    )
                    config = None
            return self._sample_and_pick_acq_best(acq) if config is None else config
        except NotPSDError as _:
            logging.warning("Chlolesky inversion failed, sampling randomly.")
            return self._get_random_config()
        except ModelFittingError as _:
            logging.warning("Botorch was unable to fit the model, sampling randomly.")
            return self._get_random_config()

    def _make_gp(self, X_tensor: Tensor, Y_tensor: Tensor) -> SingleTaskGP:
        double_precision = False
        if double_precision:
            X_tensor = X_tensor.double()
            Y_tensor = Y_tensor.double()

        noise_std = NOISE_LEVEL
        Y_tensor += noise_std * randn_like(Y_tensor)

        if self.input_warping:
            warp_tf = Warp(indices=list(range(X_tensor.shape[-1])))
        else:
            warp_tf = None
        return SingleTaskGP(X_tensor, Y_tensor, input_transform=warp_tf)

    def _config_to_feature_matrix(self, configs: List[dict]) -> Tensor:
        bounds = Tensor(self._hp_ranges.get_ndarray_bounds()).T
        X = Tensor(self._hp_ranges.to_ndarray_matrix(configs))
        return normalize(X, bounds)

    def objectives(self):
        return np.array(list(self.trial_observations.values()))

    def _sample_and_pick_acq_best(self, acq, num_samples: int = 100) -> Optional[dict]:
        """
        :param acq:
        :param num_samples:
        :return: Samples ``num_samples`` candidates and return the one maximizing
            the acquisitition function ``acq`` that was not seen earlier, if all
            samples were seen, return a random sample instead.
        """
        configs_candidates = [self._get_random_config() for _ in range(num_samples)]
        configs_candidates = [x for x in configs_candidates if x is not None]
        logger.debug(f"Sampling among {len(configs_candidates)} unseen configs")
        if configs_candidates:
            X_tensor = self._config_to_feature_matrix(configs_candidates)
            ei = acq(X_tensor.unsqueeze(dim=-2))
            return configs_candidates[ei.argmax()]
        else:
            return self._get_random_config()

    def _configs_with_results(self) -> List[dict]:
        return [
            config
            for trial, config in self.trial_configs.items()
            if not trial in self.pending_trials
        ]

    def _configs_pending(self) -> List[dict]:
        return [
            config
            for trial, config in self.trial_configs.items()
            if trial in self.pending_trials
        ]

    def metric_names(self) -> List[str]:
        return [self._metric]

    def metric_mode(self) -> str:
        return self._mode


class BotorchSearcher(BoTorchSearcher):
    """
    Downwards compatibility. Please use :class:`BoTorchSearcher` instead
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        points_to_evaluate: Optional[List[dict]] = None,
        **kwargs,
    ):
        super().__init__(
            config_space, metric, points_to_evaluate=points_to_evaluate, **kwargs
        )
        logger.warning(
            "BotorchSearcher is deprecated, please use BoTorchSearcher instead"
        )
