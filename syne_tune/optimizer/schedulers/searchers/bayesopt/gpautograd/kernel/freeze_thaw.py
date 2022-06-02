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
import autograd.numpy as anp
from autograd.builtins import isinstance
from autograd.tracer import getval

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel.base import (
    KernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel.exponential_decay import (
    ExponentialDecayResourcesKernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import (
    DEFAULT_ENCODING,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon_blocks_helpers import (
    register_parameter,
    create_encoding,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    MeanFunction,
)

__all__ = ["FreezeThawKernelFunction", "FreezeThawMeanFunction"]


class FreezeThawKernelFunction(KernelFunction):
    """
    Variant of the kernel function for modeling exponentially decaying
    learning curves, proposed in:

        Swersky, K., Snoek, J., & Adams, R. P. (2014).
        Freeze-Thaw Bayesian Optimization.
        ArXiv:1406.3896 [Cs, Stat).
        Retrieved from http://arxiv.org/abs/1406.3896

    The argument in that paper actually justifies using a non-zero mean
    function (see :class:`ExponentialDecayResourcesMeanFunction`) and
    centralizing the kernel proposed there. This is done here.

    As in the Freeze-Thaw paper, learning curves for different configs are
    conditionally independent.

    This class is configured with a kernel and a mean function over
    inputs x (dimension d) and represents a kernel (and mean function) over
    inputs (x, r) (dimension d + 1), where the resource attribute r >= 0 is
    last.

    Note: This kernel is mostly for debugging! Its conditional independence
    assumptions allow for faster inference, as implemented in
    :class:`GaussProcExpDecayPosteriorState`.
    """

    def __init__(
        self,
        kernel_x: KernelFunction,
        mean_x: MeanFunction,
        encoding_type=DEFAULT_ENCODING,
        alpha_init=1.0,
        mean_lam_init=0.5,
        gamma_init=0.5,
        max_metric_value=1.0,
        **kwargs
    ):
        """
        :param kernel_x: Kernel k_x(x, x') over configs
        :param mean_x: Mean function mu_x(x) over configs
        :param encoding_type: Encoding used for alpha, mean_lam, gamma (positive
            values)
        :param alpha_init: Initial value alpha
        :param mean_lam_init: Initial value mean_lam
        :param gamma_init: Initial value gamma
        :param max_metric_value: Maximum value which metric can attend. This is
            used as upper bound on gamma
        """
        super().__init__(dimension=kernel_x.dimension + 1, **kwargs)
        self.kernel_x = kernel_x
        self.mean_x = mean_x
        alpha_lower, alpha_upper = 1e-6, 250.0
        alpha_init = ExponentialDecayResourcesKernelFunction._wrap_initvals(
            alpha_init, alpha_lower, alpha_upper
        )
        self.encoding_alpha = create_encoding(
            encoding_type, alpha_init, alpha_lower, alpha_upper, 1, None
        )
        mean_lam_lower, mean_lam_upper = 1e-4, 50.0
        mean_lam_init = ExponentialDecayResourcesKernelFunction._wrap_initvals(
            mean_lam_init, mean_lam_lower, mean_lam_upper
        )
        self.encoding_mean_lam = create_encoding(
            encoding_type, mean_lam_init, mean_lam_lower, mean_lam_upper, 1, None
        )
        gamma_lower = max_metric_value * 0.0001
        gamma_upper = max_metric_value
        gamma_init = ExponentialDecayResourcesKernelFunction._wrap_initvals(
            gamma_init, gamma_lower, gamma_upper
        )
        self.encoding_gamma = create_encoding(
            encoding_type, gamma_init, gamma_lower, gamma_upper, 1, None
        )

        with self.name_scope():
            self.alpha_internal = register_parameter(
                self.params, "alpha", self.encoding_alpha
            )
            self.mean_lam_internal = register_parameter(
                self.params, "mean_lam", self.encoding_mean_lam
            )
            self.gamma_internal = register_parameter(
                self.params, "gamma", self.encoding_gamma
            )

    def _compute_terms(self, X, alpha, mean_lam, ret_mean=False):
        dim = self.kernel_x.dimension
        cfg = X[:, :dim]
        res = X[:, dim:]
        kappa = ExponentialDecayResourcesKernelFunction._compute_kappa(
            res, alpha, mean_lam
        )
        if ret_mean:
            mean = self.mean_x(cfg)
        else:
            mean = None

        return cfg, res, kappa, mean

    def _get_params(self, X, **kwargs):
        alpha = ExponentialDecayResourcesKernelFunction._unwrap(
            X, kwargs, "alpha", self.encoding_alpha, self.alpha_internal
        )
        mean_lam = ExponentialDecayResourcesKernelFunction._unwrap(
            X, kwargs, "mean_lam", self.encoding_mean_lam, self.mean_lam_internal
        )
        gamma = ExponentialDecayResourcesKernelFunction._unwrap(
            X, kwargs, "gamma", self.encoding_gamma, self.gamma_internal
        )

        return (alpha, mean_lam, gamma)

    @staticmethod
    def _to_tuples(cfg):
        return [
            tuple(anp.ravel(x)) for x in anp.split(cfg, getval(cfg.shape[0]), axis=0)
        ]

    def forward(self, X1, X2, **kwargs):
        alpha, mean_lam, gamma = self._get_params(X1, **kwargs)
        gamma = anp.reshape(gamma, (1, 1))
        cfg1, res1, kappa1, _ = self._compute_terms(X1, alpha, mean_lam)
        cfg1_tpls = self._to_tuples(cfg1)
        if X2 is not X1:
            cfg2, res2, kappa2, _ = self._compute_terms(X2, alpha, mean_lam)
            cfg2_tpls = self._to_tuples(cfg2)
            cfg_set = set(cfg1_tpls + cfg2_tpls)
        else:
            cfg2, res2, kappa2, cfg2_tpls = cfg1, res1, kappa1, cfg1_tpls
            cfg_set = set(cfg1_tpls)
        cfg_map = dict(zip(cfg_set, range(len(cfg_set))))
        cfg1_ind = anp.reshape(anp.array([cfg_map[x] for x in cfg1_tpls]), (-1, 1))
        if X2 is not X1:
            cfg2_ind = anp.reshape(anp.array([cfg_map[x] for x in cfg2_tpls]), (1, -1))
        else:
            cfg2_ind = anp.reshape(cfg1_ind, (1, -1))

        res2 = anp.reshape(res2, (1, -1))
        kappa2 = anp.reshape(kappa2, (1, -1))
        kappa12 = ExponentialDecayResourcesKernelFunction._compute_kappa(
            anp.add(res1, res2), alpha, mean_lam
        )
        kmat_res = anp.subtract(kappa12, anp.multiply(kappa1, kappa2))
        kmat_res = kmat_res * anp.square(gamma)
        kmat_res = kmat_res * (cfg1_ind == cfg2_ind)

        kmat_x = self.kernel_x(cfg1, cfg2)
        return kmat_x + kmat_res

    def diagonal(self, X):
        alpha, mean_lam, gamma = self._get_params(X)
        gamma = anp.reshape(gamma, (1, 1))
        cfg, res, kappa, _ = self._compute_terms(X, alpha, mean_lam)
        kappa2 = ExponentialDecayResourcesKernelFunction._compute_kappa(
            res * 2, alpha, mean_lam
        )
        kdiag_res = anp.subtract(kappa2, anp.square(kappa))
        kdiag_res = anp.reshape(kdiag_res * anp.square(gamma), (-1,))

        kdiag_x = self.kernel_x.diagonal(cfg)
        return kdiag_x + kdiag_res

    def diagonal_depends_on_X(self):
        return True

    def param_encoding_pairs(self):
        enc_list = [
            (self.alpha_internal, self.encoding_alpha),
            (self.mean_lam_internal, self.encoding_mean_lam),
            (self.gamma_internal, self.encoding_gamma),
        ]
        enc_list.extend(self.kernel_x.param_encoding_pairs())
        enc_list.extend(self.mean_x.param_encoding_pairs())
        return enc_list

    def mean_function(self, X):
        alpha, mean_lam, gamma = self._get_params(X)
        gamma = anp.reshape(gamma, (1, 1))
        cfg, res, kappa, mean = self._compute_terms(X, alpha, mean_lam, ret_mean=True)
        return anp.add(mean, anp.multiply(kappa, gamma))

    def get_params(self):
        """
        Parameter keys are alpha, mean_lam, gamma, delta (only if not fixed
        to delta_fixed_value), as well as those of self.kernel_x (prefix
        'kernelx_') and of self.mean_x (prefix 'meanx_').
        """
        values = list(self._get_params(None))
        keys = ["alpha", "mean_lam", "gamma"]
        result = {k: anp.reshape(v, (1,))[0] for k, v in zip(keys, values)}
        for pref, func in [("kernelx_", self.kernel_x), ("meanx_", self.mean_x)]:
            result.update({(pref + k): v for k, v in func.get_params().items()})
        return result

    def set_params(self, param_dict):
        for pref, func in [("kernelx_", self.kernel_x), ("meanx_", self.mean_x)]:
            len_pref = len(pref)
            stripped_dict = {
                k[len_pref:]: v for k, v in param_dict.items() if k.startswith(pref)
            }
            func.set_params(stripped_dict)
        self.encoding_alpha.set(self.alpha_internal, param_dict["alpha"])
        self.encoding_mean_lam.set(self.mean_lam_internal, param_dict["mean_lam"])
        self.encoding_gamma.set(self.gamma_internal, param_dict["gamma"])


class FreezeThawMeanFunction(MeanFunction):
    def __init__(self, kernel: FreezeThawKernelFunction, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(kernel, FreezeThawKernelFunction)
        self.kernel = kernel

    def forward(self, X):
        return self.kernel.mean_function(X)

    def param_encoding_pairs(self):
        return []

    def get_params(self):
        return dict()

    def set_params(self, param_dict):
        pass
