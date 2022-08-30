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
from typing import Callable, Optional, List
import autograd.numpy as anp
from autograd.builtins import isinstance
from numpy.random import RandomState

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import (
    DEFAULT_MCMC_CONFIG,
    MCMCConfig,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon_blocks_helpers import (
    encode_unwrap_parameter,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gp_model import (
    GaussianProcessModel,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.optimization_utils import (
    add_regularizer_to_criterion,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel import (
    KernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.likelihood import (
    GaussianProcessMarginalLikelihood,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    ScalarMeanFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_state import (
    GaussProcPosteriorState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.slice import (
    SliceSampler,
)
from syne_tune.optimizer.schedulers.utils.simple_profiler import SimpleProfiler


class GPRegressionMCMC(GaussianProcessModel):
    def __init__(
        self,
        build_kernel: Callable[[], KernelFunction],
        mcmc_config: MCMCConfig = DEFAULT_MCMC_CONFIG,
        random_seed=None,
    ):
        super().__init__(random_seed)
        self.mcmc_config = mcmc_config
        self.likelihood = _create_likelihood(
            build_kernel, random_state=self._random_state
        )
        self._states = None
        self.samples = None
        self.build_kernel = build_kernel

    @property
    def states(self) -> Optional[List[GaussProcPosteriorState]]:
        return self._states

    @property
    def number_samples(self) -> int:
        return self.mcmc_config.n_samples

    def fit(self, data: dict, profiler: Optional[SimpleProfiler] = None):
        features, targets = self._check_features_targets(
            features=data["features"], targets=data["targets"]
        )
        assert (
            targets.shape[1] == 1
        ), "targets cannot be a matrix if parameters are to be fit"

        mean_function = self.likelihood.mean
        if isinstance(mean_function, ScalarMeanFunction):
            mean_function.set_mean_value(anp.mean(targets))

        def _log_posterior_density(hp_values: anp.ndarray) -> float:
            # We check box constraints before converting hp_values to
            # internal
            if not self._is_feasible(hp_values):
                return -float("inf")
            # Decode and write into Gluon parameters
            _set_gp_hps(hp_values, self.likelihood)
            neg_log = add_regularizer_to_criterion(
                criterion=self.likelihood, crit_args=[data]
            )
            return -neg_log

        slice_sampler = SliceSampler(
            log_density=_log_posterior_density,
            scale=1.0,
            random_state=self._random_state,
        )
        init_hp_values = _get_gp_hps(self.likelihood)

        self.samples = slice_sampler.sample(
            init_hp_values,
            self.mcmc_config.n_samples,
            self.mcmc_config.n_burnin,
            self.mcmc_config.n_thinning,
        )
        self._states = self._create_posterior_states(self.samples, features, targets)

    def recompute_states(self, data: dict):
        """
        Supports fantasizing, in that targets can be a matrix. Then,
        ycols = targets.shape[1] must be a multiple of self.number_samples.

        """
        features, targets = self._check_features_targets(
            features=data["features"], targets=data["targets"]
        )
        ycols = targets.shape[1]
        if ycols > 1:
            assert ycols % self.number_samples == 0, (
                f"targets.shape[1] = {ycols}, must be multiple of number_samples"
                + f" = {self.number_samples}"
            )
        else:
            assert ycols == 1, "targets must not be empty"
        assert len(self.samples) > 0
        self._states = self._create_posterior_states(self.samples, features, targets)

    def _is_feasible(self, hp_values: anp.ndarray) -> bool:
        pos = 0
        for _, encoding in self.likelihood.param_encoding_pairs():
            lower, upper = encoding.box_constraints()
            dim = encoding.dimension
            if lower is not None or upper is not None:
                values = hp_values[pos : (pos + dim)]
                if (lower is not None) and any(values < lower):
                    return False
                if (upper is not None) and any(values > upper):
                    return False
            pos += dim
        return True

    def _create_posterior_states(self, samples, features, targets):
        ycols = targets.shape[1]
        if ycols == 1:
            num_fantasy_samples = 0
        else:
            num_fantasy_samples = ycols // self.number_samples
            ycols = num_fantasy_samples
        states = []
        offset = 0
        for sample in samples:
            likelihood = _create_likelihood(
                self.build_kernel, random_state=self._random_state
            )
            _set_gp_hps(sample, likelihood)
            targets_part = targets[:, offset : (offset + ycols)]
            state = GaussProcPosteriorState(
                features=features,
                targets=targets_part,
                mean=likelihood.mean,
                kernel=likelihood.kernel,
                noise_variance=likelihood.get_noise_variance(as_ndarray=True),
            )
            states.append(state)
            offset += num_fantasy_samples
        return states


def _get_gp_hps(likelihood: GaussianProcessMarginalLikelihood) -> anp.ndarray:
    """Get GP hyper-parameters as numpy array for a given likelihood object."""
    hp_values = []
    for param_int, encoding in likelihood.param_encoding_pairs():
        hp_values.append(encode_unwrap_parameter(param_int, encoding))
    return anp.concatenate(hp_values)


def _set_gp_hps(
    params_numpy: anp.ndarray, likelihood: GaussianProcessMarginalLikelihood
):
    """Set GP hyper-parameters from numpy array for a given likelihood object."""
    pos = 0
    for param, encoding in likelihood.param_encoding_pairs():
        dim = encoding.dimension
        values = params_numpy[pos : (pos + dim)]
        if dim == 1:
            internal_values = encoding.decode(values, param.name)
        else:
            internal_values = anp.array(
                [encoding.decode(v, param.name) for v in values]
            )
        param.set_data(internal_values)
        pos += dim


def _create_likelihood(
    build_kernel, random_state: RandomState
) -> GaussianProcessMarginalLikelihood:
    """
    Create a MarginalLikelihood object with default initial GP hyperparameters.
    """
    likelihood = GaussianProcessMarginalLikelihood(
        kernel=build_kernel(), mean=ScalarMeanFunction(), initial_noise_variance=None
    )
    # Note: The `init` parameter is a default sampler which is used only
    # for parameters which do not have initializers specified. Right now,
    # all our parameters have such initializers (constant in general),
    # so this is just to be safe (if `init` is not specified here, it
    # defaults to `np.random.uniform`, whose seed we do not control).
    likelihood.initialize(init=random_state.uniform, force_reinit=True)

    return likelihood
