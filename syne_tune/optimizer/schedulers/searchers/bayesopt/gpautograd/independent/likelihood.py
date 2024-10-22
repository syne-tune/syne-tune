from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import autograd.numpy as anp

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.likelihood import (
    MarginalLikelihood,
    GaussianProcessMarginalLikelihood,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.posterior_state import (
    IndependentGPPerResourcePosteriorState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import (
    INITIAL_NOISE_VARIANCE,
    NOISE_VARIANCE_LOWER_BOUND,
    NOISE_VARIANCE_UPPER_BOUND,
    DEFAULT_ENCODING,
    INITIAL_COVARIANCE_SCALE,
    COVARIANCE_SCALE_LOWER_BOUND,
    COVARIANCE_SCALE_UPPER_BOUND,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.distribution import (
    Gamma,
    LogNormal,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon_blocks_helpers import (
    encode_unwrap_parameter,
    register_parameter,
    create_encoding,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel import (
    KernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    MeanFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_state import (
    PosteriorState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.target_transform import (
    ScalarTargetTransform,
    IdentityTargetTransform,
)


class IndependentGPPerResourceMarginalLikelihood(MarginalLikelihood):
    r"""
    Marginal likelihood for GP multi-fidelity model over :math:`f(x, r)`,
    where for each :math:`r`, :math:`f(x, r)` is represented by an independent
    GP. The different processes share the same kernel, but have their own mean
    functions :math:`\mu_r` and covariance scales :math:`c_r`.
    If ``separate_noise_variances == True``, each process has its own noise
    variance, otherwise all processes share the same noise variance.

    :param kernel: Shared kernel function :math:`k(x, x')`
    :param mean: Maps rung level :math:`r` to mean function :math:`\mu_r`
    :param resource_attr_range: :math:`(r_{min}, r_{max})`
    :param target_transform: Invertible transform of target values y to
        latent values z, which are then modelled as Gaussian. Shared across
        different :math:`r`. Defaults to the identity
    :param separate_noise_variances: See above. Defaults to ``False``
    :param initial_noise_variance: Initial value for noise variance(s).
        Defaults to
        :const:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants.INITIAL_NOISE_VARIANCE`
    :param initial_covariance_scale: Initial value for covariance scales.
        Defaults to
        :const:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants.INITIAL_COVARIANCE_SCALE`
    :param encoding_type: Encoding used for noise variance(s) and covariance
        scales. Defaults to
        :const:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants.DEFAULT_ENCODING`
    """

    def __init__(
        self,
        kernel: KernelFunction,
        mean: Dict[int, MeanFunction],
        resource_attr_range: Tuple[int, int],
        target_transform: Optional[ScalarTargetTransform] = None,
        separate_noise_variances: bool = False,
        initial_noise_variance: Optional[float] = None,
        initial_covariance_scale: Optional[float] = None,
        encoding_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if initial_noise_variance is None:
            initial_noise_variance = INITIAL_NOISE_VARIANCE
        if initial_covariance_scale is None:
            initial_covariance_scale = INITIAL_COVARIANCE_SCALE
        if encoding_type is None:
            encoding_type = DEFAULT_ENCODING
        if target_transform is None:
            target_transform = IdentityTargetTransform()
        self.encoding_noise = create_encoding(
            encoding_name=encoding_type,
            init_val=initial_noise_variance,
            constr_lower=NOISE_VARIANCE_LOWER_BOUND,
            constr_upper=NOISE_VARIANCE_UPPER_BOUND,
            dimension=1,
            prior=Gamma(mean=0.1, alpha=0.1),
        )
        self.encoding_covscale = create_encoding(
            encoding_name=encoding_type,
            init_val=initial_covariance_scale,
            constr_lower=COVARIANCE_SCALE_LOWER_BOUND,
            constr_upper=COVARIANCE_SCALE_UPPER_BOUND,
            dimension=1,
            prior=LogNormal(0.0, 1.0),
        )
        self.mean = mean.copy()
        # Note: Child blocks in lists or dicts are not registered automatically
        for v in self.mean.values():
            self.register_child(v)
        self.kernel = kernel
        self.target_transform = target_transform
        self.resource_attr_range = resource_attr_range
        self._separate_noise_variances = separate_noise_variances
        with self.name_scope():
            self.covariance_scale_internal = {
                resource: register_parameter(
                    self.params, f"covariance_scale_{resource}", self.encoding_covscale
                )
                for resource in mean.keys()
            }
            if separate_noise_variances:
                self.noise_variance_internal = {
                    resource: register_parameter(
                        self.params, f"noise_variance_{resource}", self.encoding_noise
                    )
                    for resource in mean.keys()
                }
            else:
                self.noise_variance_internal = register_parameter(
                    self.params, "noise_variance", self.encoding_noise
                )

    def _noise_variance(self):
        if self._separate_noise_variances:
            return {
                resource: encode_unwrap_parameter(internal, self.encoding_noise)
                for resource, internal in self.noise_variance_internal.items()
            }
        else:
            return encode_unwrap_parameter(
                self.noise_variance_internal, self.encoding_noise
            )

    def _covariance_scale(self):
        return {
            resource: encode_unwrap_parameter(internal, self.encoding_covscale)
            for resource, internal in self.covariance_scale_internal.items()
        }

    def get_posterior_state(self, data: Dict[str, Any]) -> PosteriorState:
        GaussianProcessMarginalLikelihood.assert_data_entries(data)
        targets = self.target_transform(data["targets"])
        return IndependentGPPerResourcePosteriorState(
            features=data["features"],
            targets=targets,
            kernel=self.kernel,
            mean=self.mean,
            covariance_scale=self._covariance_scale(),
            noise_variance=self._noise_variance(),
            resource_attr_range=self.resource_attr_range,
        )

    def forward(self, data: Dict[str, Any]):
        return self.get_posterior_state(
            data
        ).neg_log_likelihood() + self.target_transform.negative_log_jacobian(
            data["targets"]
        )

    def _components(self) -> List[Tuple[str, MeanFunction]]:
        return [("kernel_", self.kernel), ("ytrans_", self.target_transform)] + [
            (f"mean{resource}_", mean) for resource, mean in self.mean.items()
        ]

    def param_encoding_pairs(self) -> List[tuple]:
        result = [
            x
            for _, component in self._components()
            for x in component.param_encoding_pairs()
        ]
        for internal in self.covariance_scale_internal.values():
            result.append((internal, self.encoding_covscale))
        _noise_variances = (
            list(self.noise_variance_internal.values())
            if self._separate_noise_variances
            else [self.noise_variance_internal]
        )
        for internal in _noise_variances:
            result.append((internal, self.encoding_noise))
        return result

    def get_noise_variance(self, as_ndarray=False):
        noise_variance = self._noise_variance()
        if as_ndarray:
            return noise_variance
        elif self._separate_noise_variances:
            return {k: anp.reshape(v, (1,))[0] for k, v in noise_variance.items()}
        else:
            return anp.reshape(noise_variance, (1,))[0]

    def _set_noise_variance(self, resource: int, val: float):
        if self._separate_noise_variances:
            internal = self.noise_variance_internal.get(resource)
            assert internal is not None, f"resource = {resource} not supported"
        else:
            internal = self.noise_variance_internal
        self.encoding_noise.set(internal, val)

    def get_covariance_scale(self, resource: int, as_ndarray=False):
        cov_scale = self._covariance_scale()[resource]
        return cov_scale if as_ndarray else anp.reshape(cov_scale, (1,))[0]

    def set_covariance_scale(self, resource: int, val: float):
        internal = self.covariance_scale_internal.get(resource)
        assert internal is not None, f"resource = {resource} not supported"
        self.encoding_covscale.set(internal, val)

    def get_params(self) -> Dict[str, np.ndarray]:
        result = dict()
        for pref, func in self._components():
            result.update({(pref + k): v for k, v in func.get_params().items()})
        for resource, val in self._covariance_scale().items():
            result[f"covariance_scale{resource}"] = val
        if self._separate_noise_variances:
            for resource, val in self._noise_variance().items():
                result[f"noise_variance{resource}"] = val
        else:
            result["noise_variance"] = self._noise_variance()
        return result

    def set_params(self, param_dict: Dict[str, np.ndarray]):
        for pref, func in self._components():
            len_pref = len(pref)
            stripped_dict = {
                k[len_pref:]: v for k, v in param_dict.items() if k.startswith(pref)
            }
            func.set_params(stripped_dict)
        for resource in self.covariance_scale_internal.keys():
            self.set_covariance_scale(
                resource, param_dict[f"covariance_scale{resource}"]
            )
        if self._separate_noise_variances:
            for resource in self.noise_variance_internal.keys():
                self._set_noise_variance(
                    resource, param_dict[f"noise_variance{resource}"]
                )
        else:
            self._set_noise_variance(1, param_dict["noise_variance"])

    def on_fit_start(self, data: Dict[str, Any]):
        GaussianProcessMarginalLikelihood.assert_data_entries(data)
        targets = data["targets"]
        assert (
            targets.shape[1] == 1
        ), "targets cannot be a matrix if parameters are to be fit"
        self.target_transform.on_fit_start(targets)
