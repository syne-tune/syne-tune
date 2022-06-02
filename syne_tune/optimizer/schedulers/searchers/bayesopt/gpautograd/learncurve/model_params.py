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
from typing import Dict
import autograd.numpy as anp
from autograd.tracer import getval

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import (
    COVARIANCE_SCALE_LOWER_BOUND,
    DEFAULT_ENCODING,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.distribution import (
    Gamma,
    Normal,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon_blocks_helpers import (
    encode_unwrap_parameter,
    IdentityScalarEncoding,
    register_parameter,
    create_encoding,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    MeanFunction,
)


class ISSModelParameters(MeanFunction):
    """
    Maintains parameters of an ISSM of a particular power low decay form.

    For each configuration, we have alpha < 0 and beta. These may depend
    on the input feature x (encoded configuration):
        (alpha, beta) = F(x; params),
    where params are the internal parameters to be learned.

    There is also gamma > 0, which can be fixed to 1.

    """

    def __init__(self, gamma_is_one: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.gamma_is_one = gamma_is_one
        if not gamma_is_one:
            self.gamma_encoding = create_encoding(
                DEFAULT_ENCODING,
                1.0,
                COVARIANCE_SCALE_LOWER_BOUND,
                10.0,
                1,
                Gamma(mean=1.0, alpha=0.1),
            )
            with self.name_scope():
                self.gamma_internal = register_parameter(
                    self.params, "gamma", self.gamma_encoding
                )

    def param_encoding_pairs(self):
        if self.gamma_is_one:
            return []
        else:
            return [(self.gamma_internal, self.gamma_encoding)]

    def get_gamma(self):
        if self.gamma_is_one:
            return 1.0
        else:
            gamma = encode_unwrap_parameter(self.gamma_internal, self.gamma_encoding)
            return anp.reshape(gamma, (1,))[0]

    def get_params(self):
        if self.gamma_is_one:
            return dict()
        else:
            return {"gamma": self.get_gamma()}

    def set_gamma(self, val):
        assert not self.gamma_is_one, "Cannot set gamma (fixed to 1)"
        self.gamma_encoding.set(self.gamma_internal, val)

    def set_params(self, param_dict):
        if not self.gamma_is_one:
            self.set_gamma(param_dict["gamma"])

    def get_issm_params(self, features) -> Dict:
        """
        Given feature matrix X, returns ISSM parameters which configure the
        likelihood: alpha, beta vectors (size n), gamma scalar.

        :param features: Feature matrix X, (n, d)
        :return: Dict with alpha, beta, gamma

        """
        raise NotImplementedError()


class IndependentISSModelParameters(ISSModelParameters):
    """
    Most basic implementation, where alpha, beta are scalars, independent of
    the configuration.

    """

    def __init__(self, gamma_is_one: bool = False, **kwargs):
        super().__init__(gamma_is_one, **kwargs)
        self.negalpha_encoding = create_encoding(
            DEFAULT_ENCODING, 0.5, 1e-3, 5.0, 1, Gamma(mean=0.5, alpha=0.1)
        )
        self.beta_encoding = IdentityScalarEncoding(
            constr_lower=-5.0,
            constr_upper=5.0,
            init_val=0.0,
            regularizer=Normal(0.0, 1.0),
        )
        with self.name_scope():
            self.negalpha_internal = register_parameter(
                self.params, "negalpha", self.negalpha_encoding
            )
            self.beta_internal = register_parameter(
                self.params, "beta", self.beta_encoding
            )

    def param_encoding_pairs(self):
        return super().param_encoding_pairs() + [
            (self.negalpha_internal, self.negalpha_encoding),
            (self.beta_internal, self.beta_encoding),
        ]

    def get_alpha(self):
        negalpha = encode_unwrap_parameter(
            self.negalpha_internal, self.negalpha_encoding
        )
        return -anp.reshape(negalpha, (1,))[0]

    def get_beta(self):
        beta = encode_unwrap_parameter(self.beta_internal, self.beta_encoding)
        return anp.reshape(beta, (1,))[0]

    def get_params(self):
        _params = super().get_params()
        return dict(_params, alpha=self.get_alpha(), beta=self.get_beta())

    def set_alpha(self, val):
        self.negalpha_encoding.set(self.negalpha_internal, -val)

    def set_beta(self, val):
        self.beta_encoding.set(self.beta_internal, val)

    def set_params(self, param_dict):
        super().set_params(param_dict)
        self.set_alpha(param_dict["alpha"])
        self.set_beta(param_dict["beta"])

    def get_issm_params(self, features) -> Dict:
        n = getval(features.shape[0])
        one_vec = anp.ones((n,))
        return {
            "alpha": one_vec * self.get_alpha(),
            "beta": one_vec * self.get_beta(),
            "gamma": self.get_gamma(),
        }
