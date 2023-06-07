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
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import autograd.numpy as anp
from numpy.random import RandomState

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import (
    INITIAL_NOISE_VARIANCE,
    NOISE_VARIANCE_LOWER_BOUND,
    NOISE_VARIANCE_UPPER_BOUND,
    DEFAULT_ENCODING,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.distribution import (
    Gamma,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon import Block
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon_blocks_helpers import (
    encode_unwrap_parameter,
    register_parameter,
    create_encoding,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel import (
    KernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    ScalarMeanFunction,
    MeanFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_state import (
    PosteriorState,
    GaussProcPosteriorState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.target_transform import (
    ScalarTargetTransform,
    IdentityTargetTransform,
)


class MarginalLikelihood(Block):
    """
    Interface for marginal likelihood of Gaussian-linear model.
    """

    def get_posterior_state(self, data: Dict[str, Any]) -> PosteriorState:
        raise NotImplementedError

    def forward(self, data: Dict[str, Any]):
        return self.get_posterior_state(data).neg_log_likelihood()

    def param_encoding_pairs(self) -> List[tuple]:
        """
        Return a list of tuples with the Gluon parameters of the likelihood
        and their respective encodings
        """
        raise NotImplementedError

    def box_constraints_internal(self) -> Dict[str, Tuple[float, float]]:
        """
        :return: Box constraints for all the underlying parameters
        """
        all_box_constraints = dict()
        for param, encoding in self.param_encoding_pairs():
            assert (
                encoding is not None
            ), "encoding of param {} should not be None".format(param.name)
            all_box_constraints.update(encoding.box_constraints_internal(param))
        return all_box_constraints

    def get_noise_variance(self, as_ndarray=False):
        raise NotImplementedError

    def get_params(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def set_params(self, param_dict: Dict[str, np.ndarray]):
        raise NotImplementedError

    def reset_params(self, random_state: RandomState):
        """
        Reset hyperparameters to their initial values (or resample them).
        """
        # Note: The ``init`` parameter is a default sampler which is used only
        # for parameters which do not have initializers specified. Right now,
        # all our parameters have such initializers (constant in general),
        # so this is just to be safe (if ``init`` is not specified here, it
        # defaults to ``np.random.uniform``, whose seed we do not control).
        self.initialize(init=random_state.uniform, force_reinit=True)

    def data_precomputations(self, data: Dict[str, Any], overwrite: bool = False):
        """
        Some models require precomputations based on ``data``. Precomputed
        variables are appended to ``data``. This is done only if not already
        included in ``data``, unless ``overwrite`` is True.

        :param data:
        :param overwrite:
        """
        pass

    def on_fit_start(self, data: Dict[str, Any]):
        """
        Called at the beginning of ``fit``.

        :param data: Argument passed to ``fit``

        """
        raise NotImplementedError


class GaussianProcessMarginalLikelihood(MarginalLikelihood):
    """
    Marginal likelihood of Gaussian process with Gaussian likelihood

    :param kernel: Kernel function
    :param mean: Mean function which depends on the input X only (by default,
        a scalar fitted while optimizing the likelihood)
    :param target_transform: Invertible transform of target values y to
        latent values z, which are then modelled as Gaussian. Defaults to
        the identity
    :param initial_noise_variance: A scalar to initialize the value of the
        residual noise variance
    """

    def __init__(
        self,
        kernel: KernelFunction,
        mean: Optional[MeanFunction] = None,
        target_transform: Optional[ScalarTargetTransform] = None,
        initial_noise_variance=None,
        encoding_type=None,
        **kwargs,
    ):
        super(GaussianProcessMarginalLikelihood, self).__init__(**kwargs)
        if mean is None:
            mean = ScalarMeanFunction()
        if target_transform is None:
            target_transform = IdentityTargetTransform()
        if initial_noise_variance is None:
            initial_noise_variance = INITIAL_NOISE_VARIANCE
        if encoding_type is None:
            encoding_type = DEFAULT_ENCODING
        self.encoding_noise = create_encoding(
            encoding_name=encoding_type,
            init_val=initial_noise_variance,
            constr_lower=NOISE_VARIANCE_LOWER_BOUND,
            constr_upper=NOISE_VARIANCE_UPPER_BOUND,
            dimension=1,
            prior=Gamma(mean=0.1, alpha=0.1),
        )
        self.mean = mean
        self.kernel = kernel
        self.target_transform = target_transform
        with self.name_scope():
            self.noise_variance_internal = register_parameter(
                self.params, "noise_variance", self.encoding_noise
            )

    def _noise_variance(self):
        return encode_unwrap_parameter(
            self.noise_variance_internal, self.encoding_noise
        )

    @staticmethod
    def assert_data_entries(data: Dict[str, Any]):
        features = data.get("features")
        targets = data.get("targets")
        assert (
            features is not None and targets is not None
        ), "data must contain 'features' and 'targets'"
        assert features.ndim == 2, f"features.shape = {features.shape}, must be matrix"
        if targets.ndim == 1:
            targets = targets.reshape((-1, 1))
            data["targets"] = targets
        assert features.shape[0] == targets.shape[0], (
            f"features and targets should have the same number of points "
            + f"(received {features.shape[0]} and {targets.shape[0]})"
        )

    def get_posterior_state(self, data: Dict[str, Any]) -> PosteriorState:
        self.assert_data_entries(data)
        targets = self.target_transform(data["targets"])
        return GaussProcPosteriorState(
            features=data["features"],
            targets=targets,
            mean=self.mean,
            kernel=self.kernel,
            noise_variance=self._noise_variance(),
        )

    def forward(self, data: Dict[str, Any]):
        return self.get_posterior_state(
            data
        ).neg_log_likelihood() + self.target_transform.negative_log_jacobian(
            data["targets"]
        )

    def _components(self) -> List[Tuple[str, MeanFunction]]:
        return [
            ("kernel_", self.kernel),
            ("mean_", self.mean),
            ("ytrans_", self.target_transform),
        ]

    def param_encoding_pairs(self) -> List[tuple]:
        _param_encoding_pairs = [(self.noise_variance_internal, self.encoding_noise)]
        for _, component in self._components():
            _param_encoding_pairs.extend(component.param_encoding_pairs())
        return _param_encoding_pairs

    def get_noise_variance(self, as_ndarray=False):
        noise_variance = self._noise_variance()
        return noise_variance if as_ndarray else anp.reshape(noise_variance, (1,))[0]

    def _set_noise_variance(self, val: float):
        self.encoding_noise.set(self.noise_variance_internal, val)

    def get_params(self) -> Dict[str, np.ndarray]:
        result = {"noise_variance": self.get_noise_variance()}
        for pref, func in self._components():
            result.update({(pref + k): v for k, v in func.get_params().items()})
        return result

    def set_params(self, param_dict: Dict[str, np.ndarray]):
        for pref, func in self._components():
            len_pref = len(pref)
            stripped_dict = {
                k[len_pref:]: v for k, v in param_dict.items() if k.startswith(pref)
            }
            func.set_params(stripped_dict)
        self._set_noise_variance(param_dict["noise_variance"])

    def on_fit_start(self, data: Dict[str, Any]):
        self.assert_data_entries(data)
        targets = data["targets"]
        assert (
            targets.shape[1] == 1
        ), "targets cannot be a matrix if parameters are to be fit"
        self.target_transform.on_fit_start(targets)
        targets = self.target_transform(targets)
        if isinstance(self.mean, ScalarMeanFunction):
            self.mean.set_mean_value(anp.mean(targets))
