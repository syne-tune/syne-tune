from typing import Any
import logging

import numpy as np

try:
    import torch
    from torch import Tensor, random, rand
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.transforms import Warp
    from botorch.sampling import SobolQMCNormalSampler
    from botorch.utils.transforms import normalize
    from botorch.utils.multi_objective.box_decompositions import (
        NondominatedPartitioning,
    )
    from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement
    from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
    from botorch.models.model_list_gp_regression import ModelListGP
    from botorch.optim import optimize_acqf
    from botorch.exceptions.errors import ModelFittingError
    from linear_operator.utils.errors import NotPSDError

except ImportError as e:
    logging.debug(e)

from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher

from syne_tune.optimizer.schedulers.searchers.utils import (
    make_hyperparameter_ranges,
)

logger = logging.getLogger(__name__)


class ExpectedHyperVolumeImprovement(BaseSearcher):
    """
    Implementation of expected hypervolume improvement [1] based on the BOTorch implementation.

    [1] S. Daulton, M. Balandat, and E. Bakshy.
    Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective
    Bayesian Optimization.
    Advances in Neural Information Processing Systems 33, 2020.

    :param config_space: The configuration space to sample from.
    :param points_to_evaluate: A list of configurations to evaluate initially (in the given order).
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
    :param double_precision: Whether to use double precision when fitting the GP.
    :param noise_level: Standard deviation of Gaussian noise added to observations for numerical stability.
    :param num_raw_samples: Number of raw samples to use for initialization
        when optimizing the acquisition function. Defaults to 200
    :param num_restarts: Number of restarts to use when optimizing the
        acquisition function. Defaults to 20
    :param mc_samples: Number of Monte Carlo samples to use for estimating the
        expected hypervolume improvement. Defaults to 128
    :param optimization_strategy: Strategy to optimize the acquisition function.
        Choices are "random" and "gradient" (default). If "random" is chosen
        the acquisition function is evaluated on ``num_raw_samples`` and the
        best point is returned. If "gradient" is chosen, then
        ``num_restarts`` are used with ``num_raw_samples`` each to optimize
        the acquisition function via gradient-based optimization. Defaults to "gradient"
    :param random_seed: Seed for initializing random number generators.
    """

    def __init__(
        self,
        config_space: dict[str, Any],
        points_to_evaluate: list[dict] | None = None,
        num_init_random: int = 3,
        no_fantasizing: bool = False,
        max_num_observations: int | None = 200,
        input_warping: bool = False,
        double_precision: bool = True,
        noise_level: float = 1e-8,
        num_raw_samples: int = 200,
        num_restarts: int = 20,
        mc_samples: int = 128,
        optimization_strategy: str = "gradient",
        random_seed: int = None,
    ):
        super(ExpectedHyperVolumeImprovement, self).__init__(
            config_space,
            points_to_evaluate=points_to_evaluate,
            random_seed=random_seed,
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
        self.double_precision = double_precision
        self.noise_level = noise_level
        self.num_raw_samples = num_raw_samples
        self.num_restarts = num_restarts
        self.mc_samples = mc_samples
        assert optimization_strategy in {"random", "gradient"}
        self.optimization_strategy = optimization_strategy

        # Set the random seed for botorch as well
        random.manual_seed(self.random_seed)
        self.random_state = np.random.RandomState(self.random_seed)

    def on_trial_complete(
        self, trial_id: int, config: dict[str, Any], metrics: list[float]
    ):

        self.trial_observations[trial_id] = metrics

        if trial_id in self.pending_trials:
            self.pending_trials.remove(trial_id)

    def num_suggestions(self):
        return len(self.trial_configs)

    def suggest(self) -> dict | None:
        config_suggested = self._next_points_to_evaluate()

        if config_suggested is None:
            if self.objectives().shape[0] < self.num_minimum_observations:
                config_suggested = self._get_random_config()
            else:
                config_suggested = self._sample_next_candidate()

        if config_suggested is not None:
            trial_id = len(self.trial_configs)

            # register pending
            self.pending_trials.add(trial_id)

            self.trial_configs[trial_id] = config_suggested

        return config_suggested

    def register_pending(
        self,
        trial_id: str,
        config: dict | None = None,
        milestone: int | None = None,
    ):
        self.pending_trials.add(int(trial_id))

    def evaluation_failed(self, trial_id: str):
        self.cleanup_pending(trial_id)

    def cleanup_pending(self, trial_id: str):
        trial_id = int(trial_id)
        if trial_id in self.pending_trials:
            self.pending_trials.remove(trial_id)

    def dataset_size(self):
        return len(self.trial_observations)

    def _get_gp_bounds(self):
        return Tensor(self._hp_ranges.get_ndarray_bounds()).T

    def _config_from_ndarray(self, candidate) -> dict:
        return self._hp_ranges.from_ndarray(candidate)

    def _get_random_config(self):
        return {
            k: v.sample() if hasattr(v, "sample") else v
            for k, v in self.config_space.items()
        }

    def _sample_next_candidate(self) -> dict | None:
        """
        :return: A next candidate to evaluate, if possible it is obtained by
            fitting a GP on past data and maximizing EI. If this fails because
            of numerical difficulties with non PSD matrices, then the candidate
            is sampled at random.
        """
        try:
            X = np.array(self._config_to_feature_matrix(self._configs_with_results()))
            Y = Tensor(self.objectives())
            # BoTorch only supports maximization
            Y *= -1

            if (
                self.max_num_observations is not None
                and len(X) >= self.max_num_observations
            ):
                perm = self.random_state.permutation(len(X))[
                    : self.max_num_observations
                ]
                X = X[perm]
                Y = Y[perm]
                subsample = True
            else:
                subsample = False

            X_tensor = Tensor(X)
            bounds = torch.stack([Y.min(0).values, Y.max(0).values])

            gp = self._make_gp(X_tensor=X_tensor, Y_tensor=Y)
            mll = SumMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll, max_attempts=1)

            if self.pending_trials and self.fantasising and not subsample:
                X_pending = self._config_to_feature_matrix(self._configs_pending())
            else:
                X_pending = None
            sampler = SobolQMCNormalSampler(torch.Size([self.mc_samples]))

            ref_point = torch.ones(Y.shape[1]) * 2

            partitioning = NondominatedPartitioning(ref_point=ref_point, Y=Y)
            acq_func = qLogExpectedHypervolumeImprovement(
                model=gp,
                ref_point=ref_point,  # use known reference point
                partitioning=partitioning,
                sampler=sampler,
                X_pending=X_pending,
            )

            if self.optimization_strategy == "random":
                X = rand(
                    self.num_raw_samples, len(self._hp_ranges.get_ndarray_bounds())
                )
                acq_values = acq_func(X[:, None, :])  # warm up
                best_idx = acq_values.argmax()
                candidate = X[best_idx].reshape(1, -1)
            elif self.optimization_strategy == "gradient":
                candidate, acq_value = optimize_acqf(
                    acq_func,
                    bounds=self._get_gp_bounds(),
                    q=1,
                    num_restarts=self.num_restarts,
                    raw_samples=self.num_raw_samples,
                )

            candidate = candidate.detach().numpy()[0]
            return self._config_from_ndarray(candidate)

        except NotPSDError as _:
            logging.warning("Chlolesky inversion failed, sampling randomly.")
            return self._get_random_config()
        except ModelFittingError as _:
            logging.warning("Botorch was unable to fit the model, sampling randomly.")
            return self._get_random_config()

    #        except:
    # BoTorch can raise different errors, easier to not try to catch them individually
    #            logging.warning("Botorch was unable to fit the model, sampling randomly.")
    #            return self._get_random_config()

    def _make_gp(self, X_tensor, Y_tensor):

        if self.double_precision:
            X_tensor = X_tensor.double()
            Y_tensor = Y_tensor.double()

        if self.input_warping:
            warp_tf = Warp(indices=list(range(X_tensor.shape[-1])))
        else:
            warp_tf = None

        models = []
        for i in range(Y_tensor.shape[-1]):
            train_y = Y_tensor[..., i : i + 1]
            noise_y = torch.full_like(train_y, self.noise_level)

            models.append(
                SingleTaskGP(X_tensor, train_y, noise_y, input_transform=warp_tf)
            )
        model = ModelListGP(*models)
        return model

    def _config_to_feature_matrix(self, configs: list[dict]):
        bounds = Tensor(self._hp_ranges.get_ndarray_bounds()).T
        X = Tensor(self._hp_ranges.to_ndarray_matrix(configs))
        return normalize(X, bounds)

    def objectives(self):
        return np.array(list(self.trial_observations.values()))

    def _configs_with_results(self) -> list[dict]:
        return [
            config
            for trial, config in self.trial_configs.items()
            if not trial in self.pending_trials
        ]

    def _configs_pending(self) -> list[dict]:
        return [
            config
            for trial, config in self.trial_configs.items()
            if trial in self.pending_trials
        ]
