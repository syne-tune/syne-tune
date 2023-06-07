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
from typing import Dict, Any
import autograd.numpy as anp
from autograd.tracer import getval

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    MeanFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon_blocks_helpers import (
    IdentityScalarEncoding,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.distribution import (
    Normal,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon_blocks_helpers import (
    encode_unwrap_parameter,
    register_parameter,
)


class ScalarTargetTransform(MeanFunction):
    r"""
    Interface for invertible transforms of scalar target values.

    :meth:`forward` maps original target values :math:`y` to latent target values
    :math:`z`, the latter are typically modelled as Gaussian.
    :meth:`negative_log_jacobian` returns the term to be added to :math:`-\log P(z)`,
    where :math:`z` is mapped from :math:`y`, in order to obtain :math:`-\log P(y)`.
    """

    def forward(self, targets):
        """
        :param targets: Target vector :math:`y` in original form
        :return: Transformed latent target vector :math:`z`
        """
        raise NotImplementedError

    def negative_log_jacobian(self, targets):
        r"""
        :param targets: Target vector :math:`y` in original form
        :return: Term to add to :math:`-\log P(z)` to obtain :math:`-\log P(y)`
        """
        raise NotImplementedError

    def inverse(self, latents):
        """
        :param latents: Latent target vector :math:`z`
        :return: Corresponding target vector :math:`y`
        """
        raise NotImplementedError

    def on_fit_start(self, targets):
        """
        This is called just before the surrogate model optimization starts.

        :param targets: Target vector :math:`y` in original form
        """
        pass


class IdentityTargetTransform(ScalarTargetTransform):
    def forward(self, targets):
        return targets

    def negative_log_jacobian(self, targets):
        return 0.0

    def inverse(self, latents):
        return latents

    def param_encoding_pairs(self):
        return []

    def get_params(self) -> Dict[str, Any]:
        return dict()

    def set_params(self, param_dict: Dict[str, Any]):
        pass


BOXCOX_LAMBDA_LOWER_BOUND = -1.0

BOXCOX_LAMBDA_UPPER_BOUND = 2.0

BOXCOX_LAMBDA_INITVAL = 0.5

BOXCOX_LAMBDA_NAME = "boxcox_lambda"

BOXCOX_LAMBDA_EPS = 1e-7

BOXCOX_ZLAMBDA_THRES = -1.0 + 1e-10

BOXCOX_TARGET_THRES = 1e-10

BOXCOX_LAMBDA_OPT_MIN_NUMDATA = 5


class BoxCoxTargetTransform(ScalarTargetTransform):
    r"""
    The Box-Cox transform for :math:`y > 0` is parameterized in terms of
    :math:`\lambda`:

    .. math::

       z = T(y, \lambda) = \frac{y^{\lambda} - 1}{\lambda},\quad \lambda\ne 0

       T(y, \lambda=0) = \log y

    One difficulty is that expressions involve division by :math:`\lambda`. Our
    implementation separates between (1) :math:`\lambda \ge \varepsilon`, (2)
    :math:`\lambda\le -\varepsilon`, and (3)
    :math:`-\varepsilon < \lambda < \varepsilon`, where :math:`\varepsilon` is
    :const:`BOXCOX_LAMBDA_EPS`. In case (3), we use the approximation
    :math:`z \approx u + \lambda u^2/2`, where :math:`u = \log y`.

    Note that we require :math:`1 + z\lambda > 0`, which restricts :math:`z` if
    :math:`\lambda\ne 0`.

    .. note::
       Targets must be positive. They are thresholded at
       :const:`BOXCOX_TARGET_THRES`, so negative targets do not raise an error.

    The Box-Cox transform has been proposed in the content of Bayesian optimization
    by

        | Cowen-Rivers, A. et.al.
        | HEBO: Pushing the Limits of Sample-efficient Hyper-parameter Optimisation
        | Journal of Artificial Intelligence Research 74 (2022), 1269-1349
        | `ArXiV <https://arxiv.org/abs/2012.03826>`__

    However, they decouple the transformation of targets from fitting the remaining
    surrogate model parameters, which is possible only under a simplifying
    assumption (namely, that targets after transform are modelled i.i.d. by a
    single univariate Gaussian). Instead, we treat :math:`\lambda` as just one
    more parameter to fit along with all the others.
    """

    def __init__(
        self,
        initial_boxcox_lambda=None,
        **kwargs,
    ):
        super(BoxCoxTargetTransform, self).__init__(**kwargs)
        if initial_boxcox_lambda is None:
            initial_boxcox_lambda = BOXCOX_LAMBDA_INITVAL
        # Normal prior is such that feasible range equals to +- 2 sigma
        self.encoding = IdentityScalarEncoding(
            init_val=initial_boxcox_lambda,
            constr_lower=BOXCOX_LAMBDA_LOWER_BOUND,
            constr_upper=BOXCOX_LAMBDA_UPPER_BOUND,
            dimension=1,
            regularizer=Normal(BOXCOX_LAMBDA_INITVAL, 0.75),
        )
        self.encoding_fixed = IdentityScalarEncoding(
            init_val=initial_boxcox_lambda,
            constr_lower=initial_boxcox_lambda,
            constr_upper=initial_boxcox_lambda,
            dimension=1,
        )
        self._boxcox_lambda_fixed = False
        with self.name_scope():
            self.boxcox_lambda_internal = register_parameter(
                self.params, BOXCOX_LAMBDA_NAME, self.encoding
            )

    def _current_encoding(self):
        return self.encoding_fixed if self._boxcox_lambda_fixed else self.encoding

    def param_encoding_pairs(self):
        return [(self.boxcox_lambda_internal, self._current_encoding())]

    def _get_boxcox_lambda(self):
        return encode_unwrap_parameter(
            self.boxcox_lambda_internal, self._current_encoding()
        )

    def get_boxcox_lambda(self):
        return self._get_boxcox_lambda()[0]

    def set_boxcox_lambda(self, boxcox_lambda):
        self.encoding.set(self.boxcox_lambda_internal, boxcox_lambda)

    def get_params(self) -> Dict[str, Any]:
        return {BOXCOX_LAMBDA_NAME: self.get_boxcox_lambda()}

    def set_params(self, param_dict: Dict[str, Any]):
        if not self._boxcox_lambda_fixed:
            self.set_boxcox_lambda(param_dict[BOXCOX_LAMBDA_NAME])

    def _get_uvals(self, targets):
        return anp.log(anp.maximum(targets, BOXCOX_TARGET_THRES))

    def negative_log_jacobian(self, targets):
        boxcox_lambda = self._get_boxcox_lambda()
        u_mean = anp.mean(self._get_uvals(targets))
        return u_mean * (1.0 - boxcox_lambda)

    # Case (1)
    def _forward_lam_gt_eps(self, numerator, boxcox_lambda):
        return anp.divide(numerator, anp.maximum(boxcox_lambda, BOXCOX_LAMBDA_EPS))

    # Case (2)
    def _forward_lam_lt_minuseps(self, numerator, boxcox_lambda):
        return anp.divide(numerator, anp.minimum(boxcox_lambda, -BOXCOX_LAMBDA_EPS))

    # Case (3)
    def _forward_abslam_lt_eps(self, uvals, boxcox_lambda):
        return anp.multiply(uvals, 0.5 * uvals * boxcox_lambda + 1.0)

    def forward(self, targets):
        uvals = self._get_uvals(targets)
        ndim = getval(targets.ndim)
        boxcox_lambda = anp.reshape(self._get_boxcox_lambda(), (1,) * ndim)
        # Case distinction, in order to avoid division by (almost) zero
        numerator = anp.expm1(uvals * boxcox_lambda)
        return anp.where(
            boxcox_lambda >= BOXCOX_LAMBDA_EPS,
            self._forward_lam_gt_eps(numerator, boxcox_lambda),
            anp.where(
                boxcox_lambda <= -BOXCOX_LAMBDA_EPS,
                self._forward_lam_lt_minuseps(numerator, boxcox_lambda),
                self._forward_abslam_lt_eps(uvals, boxcox_lambda),
            ),
        )

    def _inverse_lam_gt_eps(self, log_1_plus_z_lambda, boxcox_lambda):
        return anp.exp(
            anp.divide(
                log_1_plus_z_lambda, anp.maximum(boxcox_lambda, BOXCOX_LAMBDA_EPS)
            )
        )

    def _inverse_lam_lt_minuseps(self, log_1_plus_z_lambda, boxcox_lambda):
        return anp.exp(
            anp.divide(
                log_1_plus_z_lambda, anp.minimum(boxcox_lambda, -BOXCOX_LAMBDA_EPS)
            )
        )

    def _inverse_abslam_lt_eps(self, z_lambda, z):
        return anp.exp(anp.multiply(z, 1.0 - 0.5 * z_lambda))

    def inverse(self, latents):
        r"""
        The inverse is :math:`\exp( \log(1 + z\lambda) / \lambda )`. For
        :math:`\lambda\approx 0`, we use :math:`\exp( z (1 - z\lambda/2) )`.

        We also need :math:`1 + z\lambda > 0`, so we use the maximum of
        :math:`z lambda` and :const:`BOXCOX_ZLAMBDA_THRES`.
        """
        ndim = getval(latents.ndim)
        boxcox_lambda = anp.reshape(self._get_boxcox_lambda(), (1,) * ndim)
        z_lambda = anp.maximum(latents * boxcox_lambda, BOXCOX_ZLAMBDA_THRES)
        log_1_plus_z_lambda = anp.log1p(z_lambda)
        return anp.where(
            boxcox_lambda >= BOXCOX_LAMBDA_EPS,
            self._inverse_lam_gt_eps(log_1_plus_z_lambda, boxcox_lambda),
            anp.where(
                boxcox_lambda <= -BOXCOX_LAMBDA_EPS,
                self._inverse_lam_lt_minuseps(log_1_plus_z_lambda, boxcox_lambda),
                self._inverse_abslam_lt_eps(z_lambda, latents),
            ),
        )

    def on_fit_start(self, targets):
        """
        We only optimize ``boxcox_lambda`` once there are no less than
        :const:`BOXCOX_LAMBDA_OPT_MIN_NUMDATA` data points. Otherwise, it remains
        fixed to its initial value.
        """
        self._boxcox_lambda_fixed = targets.size < BOXCOX_LAMBDA_OPT_MIN_NUMDATA
