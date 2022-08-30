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
from autograd.tracer import getval

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import (
    INITIAL_COVARIANCE_SCALE,
    INITIAL_INVERSE_BANDWIDTHS,
    DEFAULT_ENCODING,
    INVERSE_BANDWIDTHS_LOWER_BOUND,
    INVERSE_BANDWIDTHS_UPPER_BOUND,
    COVARIANCE_SCALE_LOWER_BOUND,
    COVARIANCE_SCALE_UPPER_BOUND,
    NUMERICAL_JITTER,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.distribution import (
    Uniform,
    LogNormal,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon import Block
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon_blocks_helpers import (
    encode_unwrap_parameter,
    register_parameter,
    create_encoding,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    MeanFunction,
)

__all__ = ["KernelFunction", "Matern52"]


class KernelFunction(MeanFunction):
    """
    Base class of kernel (or covariance) functions
    """

    def __init__(self, dimension: int, **kwargs):
        """
        :param dimension: Dimensionality of input points after encoding into
            ndarray
        """
        super().__init__(**kwargs)
        self._dimension = dimension

    @property
    def dimension(self):
        """
        :return: Dimension d of input points
        """
        return self._dimension

    def diagonal(self, X):
        """
        :param X: Input data, shape (n, d)
        :return: Diagonal of K(X, X), shape (n,)
        """
        raise NotImplementedError

    def diagonal_depends_on_X(self):
        """
        For stationary kernels, diagonal does not depend on X

        :return: Does diagonal(X) depend on X?
        """
        raise NotImplementedError

    def _check_input_shape(self, X):
        return anp.reshape(X, (getval(X.shape[0]), self._dimension))


class SquaredDistance(Block):
    """
    Block that is responsible for the computation of matrices of squared
    distances. The distances can possibly be weighted (e.g., ARD
    parametrization). For instance:
        X1 with size (n1,d)
        X2 with size (n2,d)
        inverse_bandwidths with size (1,d)
        results in a matrix of size (n1,n2) with i,j entry equal to
            sum_{k=1}^d (X1[i,k] - X2[j,k])^2 * inverse_bandwidths[k]^2

    if ARD == False, inverse_bandwidths is equal to a scalar broadcast to the
    d components (with d=dimension, i.e., the number of features in X)
    otherwise, inverse_bandwidths is (1,d)
    """

    def __init__(self, dimension, ARD=False, encoding_type=DEFAULT_ENCODING, **kwargs):
        super().__init__(**kwargs)
        self.ARD = ARD
        inverse_bandwidths_dimension = 1 if not ARD else dimension
        self.encoding = create_encoding(
            encoding_type,
            INITIAL_INVERSE_BANDWIDTHS,
            INVERSE_BANDWIDTHS_LOWER_BOUND,
            INVERSE_BANDWIDTHS_UPPER_BOUND,
            inverse_bandwidths_dimension,
            Uniform(INVERSE_BANDWIDTHS_LOWER_BOUND, INVERSE_BANDWIDTHS_UPPER_BOUND),
        )

        with self.name_scope():
            self.inverse_bandwidths_internal = register_parameter(
                self.params,
                "inverse_bandwidths",
                self.encoding,
                shape=(inverse_bandwidths_dimension,),
            )

    def _inverse_bandwidths(self):
        return encode_unwrap_parameter(self.inverse_bandwidths_internal, self.encoding)

    def forward(self, X1, X2):
        """
        Actual computation of the matrix of squared distances (see details above)

        :param X1: input data of size (n1,d)
        :param X2: input data of size (n2,d)
        """
        # In case inverse_bandwidths if of size (1, dimension), dimension>1,
        # ARD is handled by broadcasting
        inverse_bandwidths = anp.reshape(self._inverse_bandwidths(), (1, -1))

        if X2 is X1:
            X1_scaled = anp.multiply(X1, inverse_bandwidths)
            D = -2.0 * anp.dot(X1_scaled, anp.transpose(X1_scaled))
            X1_squared_norm = anp.sum(anp.square(X1_scaled), axis=1)
            D = D + anp.reshape(X1_squared_norm, (1, -1))
            D = D + anp.reshape(X1_squared_norm, (-1, 1))
        else:
            X1_scaled = anp.multiply(X1, inverse_bandwidths)
            X2_scaled = anp.multiply(X2, inverse_bandwidths)
            X1_squared_norm = anp.sum(anp.square(X1_scaled), axis=1)
            X2_squared_norm = anp.sum(anp.square(X2_scaled), axis=1)
            D = -2.0 * anp.matmul(X1_scaled, anp.transpose(X2_scaled))
            D = D + anp.reshape(X1_squared_norm, (-1, 1))
            D = D + anp.reshape(X2_squared_norm, (1, -1))

        return anp.abs(D)

    def get_params(self):
        """
        Parameter keys are inv_bw<k> if dimension > 1, and inv_bw if
        dimension == 1.
        """
        inverse_bandwidths = anp.reshape(self._inverse_bandwidths(), (-1,))
        if inverse_bandwidths.size == 1:
            return {"inv_bw": inverse_bandwidths[0]}
        else:
            return {
                "inv_bw{}".format(k): inverse_bandwidths[k]
                for k in range(inverse_bandwidths.size)
            }

    def set_params(self, param_dict):
        dimension = self.encoding.dimension
        if dimension == 1:
            inverse_bandwidths = [param_dict["inv_bw"]]
        else:
            keys = ["inv_bw{}".format(k) for k in range(dimension)]
            for k in keys:
                assert k in param_dict, "'{}' not in param_dict = {}".format(
                    k, param_dict
                )
            inverse_bandwidths = [param_dict[k] for k in keys]
        self.encoding.set(self.inverse_bandwidths_internal, inverse_bandwidths)


class Matern52(KernelFunction):
    """
    Block that is responsible for the computation of Matern52 kernel
    matrices. For instance:
        X1 with size (n1,d)
        X2 with size (n2,d)
    results in a matrix of size (n1,n2) with i,j entry equal to the
    Matern52 kernel at (X1[i,:], X2[j,:]).

    If ARD == False, inverse_bandwidths is equal to a scalar broadcast to the
    d components (with d=dimension, i.e., the number of features in X)
    otherwise (ARD == True), inverse_bandwidths is (1,d)
    """

    def __init__(
        self,
        dimension,
        ARD=False,
        encoding_type=DEFAULT_ENCODING,
        has_covariance_scale=True,
        **kwargs
    ):
        super(Matern52, self).__init__(dimension, **kwargs)
        self.ARD = ARD
        self.has_covariance_scale = has_covariance_scale
        self.squared_distance = SquaredDistance(
            dimension=dimension, ARD=ARD, encoding_type=encoding_type
        )
        if has_covariance_scale:
            self.encoding = create_encoding(
                encoding_name=encoding_type,
                init_val=INITIAL_COVARIANCE_SCALE,
                constr_lower=COVARIANCE_SCALE_LOWER_BOUND,
                constr_upper=COVARIANCE_SCALE_UPPER_BOUND,
                dimension=1,
                prior=LogNormal(0.0, 1.0),
            )
            with self.name_scope():
                self.covariance_scale_internal = register_parameter(
                    self.params, "covariance_scale", self.encoding
                )

    def _covariance_scale(self):
        if self.has_covariance_scale:
            return encode_unwrap_parameter(
                self.covariance_scale_internal, self.encoding
            )
        else:
            return 1.0

    def forward(self, X1, X2):
        """
        Actual computation of the Matern52 kernel matrix (see details above)

        :param X1: input data of size (n1, d)
        :param X2: input data of size (n2, d)
        """
        covariance_scale = self._covariance_scale()
        X1 = self._check_input_shape(X1)
        if X2 is not X1:
            X2 = self._check_input_shape(X2)
        D = self.squared_distance(X1, X2)
        # Using the plain np.sqrt is numerically unstable for D ~ 0
        # (non-differentiability)
        # that's why we add NUMERICAL_JITTER
        B = anp.sqrt(5.0 * D + NUMERICAL_JITTER)
        K = anp.multiply((1.0 + B + 5.0 / 3.0 * D) * anp.exp(-B), covariance_scale)

        return K

    def diagonal(self, X):
        X = self._check_input_shape(X)
        covariance_scale = self._covariance_scale()
        covariance_scale_times_ones = anp.multiply(
            anp.ones((getval(X.shape[0]), 1)), covariance_scale
        )

        return anp.reshape(covariance_scale_times_ones, (-1,))

    def diagonal_depends_on_X(self):
        return False

    def param_encoding_pairs(self):
        result = [
            (
                self.squared_distance.inverse_bandwidths_internal,
                self.squared_distance.encoding,
            )
        ]
        if self.has_covariance_scale:
            result.insert(0, (self.covariance_scale_internal, self.encoding))
        return result

    def get_covariance_scale(self):
        if self.has_covariance_scale:
            return self._covariance_scale()[0]
        else:
            return 1.0

    def set_covariance_scale(self, covariance_scale):
        assert self.has_covariance_scale, "covariance_scale is fixed to 1"
        self.encoding.set(self.covariance_scale_internal, covariance_scale)

    def get_params(self):
        result = self.squared_distance.get_params()
        if self.has_covariance_scale:
            result["covariance_scale"] = self.get_covariance_scale()
        return result

    def set_params(self, param_dict):
        self.squared_distance.set_params(param_dict)
        if self.has_covariance_scale:
            self.set_covariance_scale(param_dict["covariance_scale"])
