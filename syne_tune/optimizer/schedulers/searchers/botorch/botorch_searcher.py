from typing import Optional, List, Dict, Any
import logging

import numpy as np

from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import (
    SingleObjectiveBaseSearcher,
)
from syne_tune.optimizer.schedulers.searchers.utils import (
    make_hyperparameter_ranges,
)


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


logger = logging.getLogger(__name__)


NOISE_LEVEL = 1e-3


class BoTorchSearcher(SingleObjectiveBaseSearcher):
    """
    A searcher that suggest configurations using BOTORCH to build GP surrogate
    and optimize acquisition function.

    ``qExpectedImprovement`` is used for the acquisition function, given that it
    supports pending evaluations.

    :param config_space: Configuration space for the evaluation function.
    :param points_to_evaluate: A set of initial configurations to be evaluated before starting the optimization.
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
     :param random_seed: Seed for initializing random number generators.
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        points_to_evaluate: Optional[List[dict]] = None,
        num_init_random: int = 3,
        no_fantasizing: bool = False,
        max_num_observations: Optional[int] = 200,
        input_warping: bool = True,
        random_seed: int = None,
    ):
        super(BoTorchSearcher, self).__init__(
            config_space, points_to_evaluate=points_to_evaluate, random_seed=random_seed
        )
        assert num_init_random >= 2
        self.num_minimum_observations = num_init_random
        self.fantasising = not no_fantasizing
        self.max_num_observations = max_num_observations
        self.input_warping = input_warping
        self.trial_configs = dict()
        self.pending_trials = set()
        self.trial_observations = dict()

        self._hp_ranges = make_hyperparameter_ranges(config_space)

        # Set the random seed for botorch as well
        random.manual_seed(self.random_seed)
        self.random_state = np.random.RandomState(self.random_seed)

    def on_trial_complete(
        self,
        trial_id: int,
        config: Dict[str, Any],
        metric: float,
    ):
        trial_id = int(trial_id)
        self.trial_observations[trial_id] = metric
        if trial_id in self.pending_trials:
            self.pending_trials.remove(trial_id)

    def num_suggestions(self):
        return len(self.trial_configs)

    def _get_random_config(self):
        return {
            k: v.sample() if hasattr(v, "sample") else v
            for k, v in self.config_space.items()
        }

    def suggest(self) -> Optional[dict]:

        config_suggested = self._next_points_to_evaluate()

        if config_suggested is None:
            if len(self.objectives()) < self.num_minimum_observations:
                config_suggested = self._get_random_config()
            else:
                config_suggested = self._sample_next_candidate()

        if config_suggested is not None:
            # assign new internal trial_id
            trial_id = len(self.trial_configs)

            # register pending
            self.pending_trials.add(trial_id)

            self.trial_configs[trial_id] = config_suggested

        return config_suggested

    def evaluation_failed(self, trial_id: int):
        self.cleanup_pending(trial_id)

    def cleanup_pending(self, trial_id: int):
        if trial_id in self.pending_trials:
            self.pending_trials.remove(trial_id)

    def dataset_size(self):
        return len(self.trial_observations)

    def _get_gp_bounds(self):
        return Tensor(self._hp_ranges.get_ndarray_bounds()).T

    def _config_from_ndarray(self, candidate) -> dict:
        return self._hp_ranges.from_ndarray(candidate)

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

            # Continuous optimization of acquisition function only if
            # ``restrict_configurations`` not used
            candidate, acq_value = optimize_acqf(
                acq,
                bounds=self._get_gp_bounds(),
                q=1,
                num_restarts=3,
                raw_samples=100,
            )
            candidate = candidate.detach().numpy()[0]
            return self._config_from_ndarray(candidate)

        except NotPSDError as _:
            logging.warning("Chlolesky inversion failed, sampling randomly.")
            return self._get_random_config()
        except ModelFittingError as _:
            logging.warning("Botorch was unable to fit the model, sampling randomly.")
            return self._get_random_config()
        except:
            # BoTorch can raise different errors, easier to not try to catch them individually
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
