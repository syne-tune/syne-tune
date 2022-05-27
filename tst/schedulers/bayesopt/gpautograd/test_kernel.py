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
import pytest

import numpy as np
import autograd.numpy as anp
from autograd.tracer import getval

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel \
    import Matern52, FabolasKernelFunction, ProductKernelFunction, \
    CrossValidationKernelFunction, KernelFunction
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean \
    import ZeroMeanFunction
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel.base \
    import SquaredDistance
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants \
    import DATA_TYPE
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon_blocks_helpers \
    import LogarithmScalarEncoding, PositiveScalarEncoding
from syne_tune.config_space import uniform
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory \
    import make_hyperparameter_ranges
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.config_ext \
    import ExtendedConfiguration


def test_square_distance_no_ard_unit_bandwidth():
    X = anp.array([[1, 0], [0, 1]], dtype=DATA_TYPE)
    # test default ard = False
    sqd = SquaredDistance(dimension=2)
    assert sqd.ARD == False
    sqd.collect_params().initialize()
    D = sqd(X, X)
    expected_D = anp.array([[0.0, 2.0], [2.0, 0.0]])
    np.testing.assert_almost_equal(expected_D, D)
    

def test_square_distance_no_ard_non_unit_bandwidth():
    X = anp.array([[1, 0], [0, 1]], dtype=DATA_TYPE)
    sqd = SquaredDistance(dimension=2)
    assert sqd.ARD == False
    sqd.collect_params().initialize()
    sqd.encoding.set(sqd.inverse_bandwidths_internal, 1./anp.sqrt(2.))
    D = sqd(X, X)
    expected_D = anp.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_almost_equal(expected_D, D)
    

def test_square_distance_with_ard():
    X = anp.array([[2., 1.], [1., 2.], [0., 1.]], dtype=DATA_TYPE)
    sqd = SquaredDistance(dimension=2, ARD=True)
    assert sqd.ARD == True
    sqd.collect_params().initialize()
    sqd.encoding.set(sqd.inverse_bandwidths_internal, [1. / anp.sqrt(2.), 1.])
    D = sqd(X, X)
    expected_D = anp.array([[0., 3./2., 2.], [3./2., 0., 3./2.], [2.0, 3./2., 0.]])
    np.testing.assert_almost_equal(expected_D, D)
    

mater52 = lambda squared_dist: \
    (1. + anp.sqrt(5. * squared_dist) +
     5. / 3. * squared_dist) * anp.exp(-anp.sqrt(5. * squared_dist))
freeze_thaw = lambda u, alpha, beta: beta**alpha / (u + beta)**alpha


def test_matern52_unit_scale():
    X = anp.array([[1, 0], [0, 1]], dtype=DATA_TYPE)
    kernel = Matern52(dimension=2)
    assert kernel.ARD == False
    kernel.collect_params().initialize()
    K = kernel(X,X)
    expected_K = anp.array([[mater52(0.0), mater52(2.0)], [mater52(2.0), mater52(0.0)]])
    np.testing.assert_almost_equal(expected_K, K)
    

def test_matern52_non_unit_scale():
    X = anp.array([[1, 0], [0, 1]], dtype=DATA_TYPE)
    kernel = Matern52(dimension=2)
    assert kernel.ARD == False
    kernel.collect_params().initialize()
    kernel.encoding.set(kernel.covariance_scale_internal, 0.5)
    K = kernel(X,X)
    expected_K = 0.5 * anp.array([[mater52(0.0), mater52(2.0)], [mater52(2.0), mater52(0.0)]])
    np.testing.assert_almost_equal(expected_K, K)
    

def test_matern52_ard():
    X = anp.array([[2., 1.], [1., 2.], [0., 1.]], dtype=DATA_TYPE)
    kernel = Matern52(dimension=2, ARD=True)
    kernel.collect_params().initialize()
    sqd = kernel.squared_distance
    assert kernel.ARD == True
    assert sqd.ARD == True
    sqd.encoding.set(sqd.inverse_bandwidths_internal, [1. / anp.sqrt(2.), 1.])
    K = kernel(X,X)
    # expected_D is taken from previous test about squared distances
    expected_D = anp.array([[0., 3. / 2., 2.], [3. / 2., 0., 3. / 2.], [2.0, 3. / 2., 0.]])
    expected_K = mater52(expected_D)
    np.testing.assert_almost_equal(expected_K, K)


def test_matern52_encoding():
    kernel = Matern52(dimension=2, ARD=True)
    assert isinstance(kernel.encoding, LogarithmScalarEncoding)
    assert isinstance(kernel.squared_distance.encoding, LogarithmScalarEncoding)
    assert kernel.encoding.dimension == 1
    assert kernel.squared_distance.encoding.dimension == 2
    kernel = Matern52(dimension=2, ARD=True, encoding_type="positive")
    assert isinstance(kernel.encoding, PositiveScalarEncoding)
    assert isinstance(kernel.squared_distance.encoding, PositiveScalarEncoding)
    assert kernel.encoding.dimension == 1
    assert kernel.squared_distance.encoding.dimension == 2
    

def test_fabolas_encoding():
    kernel = FabolasKernelFunction()
    assert isinstance(kernel.encoding_u12, LogarithmScalarEncoding)
    assert kernel.encoding_u12.dimension == 1

    kernel = FabolasKernelFunction(encoding_type="positive")
    assert isinstance(kernel.encoding_u12, PositiveScalarEncoding)
    assert kernel.encoding_u12.dimension == 1


def test_matern52_wrongshape():
    kernel = Matern52(dimension=3)
    kernel.collect_params().initialize()
    X1 = anp.random.normal(0.0, 1.0, (5, 2))

    with pytest.raises(Exception):
        kernel(X1, X1)

    with pytest.raises(Exception):
        kernel.diagonal(X1)

    X2 = anp.random.normal(0.0, 1.0, (3, 3))
    with pytest.raises(Exception):
        kernel(X2, X1)


def _create_product_kernel(kernel1, kernel2):
    kernel1.collect_params().initialize()
    kernel2.collect_params().initialize()
    return ProductKernelFunction(kernel1, kernel2)


@pytest.mark.parametrize('kernel1, kernel2, input_data_dimension', [
    (Matern52(dimension=1), Matern52(dimension=1), 2),
    (Matern52(dimension=2), Matern52(dimension=2), 4),
    (Matern52(dimension=3), Matern52(dimension=1), 4),
    (FabolasKernelFunction(), Matern52(dimension=2), 3),
])
@pytest.mark.parametrize('X0_samples, X1_samples', [
    (1, 1),
    (1, 2),
    (5, 7)
])
def test_product_kernel_happy_path(kernel1, kernel2, input_data_dimension, X0_samples, X1_samples):
    product_kernel = _create_product_kernel(kernel1, kernel2)
    X0 = anp.random.randn(X0_samples, input_data_dimension)
    X1 = anp.random.randn(X1_samples, input_data_dimension)
    kernel_output = product_kernel(X0, X1)
    assert kernel_output.shape == (X0_samples, X1_samples)


@pytest.mark.parametrize('kernel1, kernel2', [
    (Matern52(dimension=2), Matern52(dimension=1)),
    (FabolasKernelFunction(), Matern52(dimension=2)),
])
def test_product_kernel_wrong_shape(kernel1, kernel2):
    product_kernel = _create_product_kernel(kernel1, kernel2)

    X1 = anp.random.randn(5, 4)
    with pytest.raises(Exception):
        product_kernel(X1, X1)
    
    with pytest.raises(Exception):
        product_kernel.diagonal(X1)

    X2 = anp.random.randn(3, 3)
    with pytest.raises(Exception):
        product_kernel(X2, X1)

    X3 = anp.random.randn(5, 2)
    with pytest.raises(Exception):
        product_kernel(X3, X3)


class ConstantKernelFunction(KernelFunction):
    """
    Kernel which is constant 1
    """
    def __init__(self, dimension=1, **kwargs):
        super().__init__(dimension=dimension, **kwargs)

    def diagonal(self, X):
        X = self._check_input_shape(X)
        return anp.ones((getval(X.shape[0]),))

    def diagonal_depends_on_X(self):
        return False

    def forward(self, X1, X2):
        X1 = self._check_input_shape(X1)
        if X2 is not X1:
            X2 = self._check_input_shape(X2)
        return anp.ones((getval(X1.shape[0]), getval(X2.shape[0])))

    def param_encoding_pairs(self):
        return []

    def get_params(self):
        return dict()

    def set_params(self, param_dict):
        pass


def test_crossvalidation_kernel():
    num_folds = 6
    config_space = {'x': uniform(0.0, 1.0)}
    hp_ranges = make_hyperparameter_ranges(config_space)
    config_ext = ExtendedConfiguration(
        hp_ranges, resource_attr_key='epoch',
        resource_attr_range=(1, num_folds))
    num_configs = 50
    xvals = np.random.rand(num_configs)
    rvals = np.random.randint(low=1, high=num_folds + 1, size=num_configs)
    configs = []
    for x, r in zip(xvals, rvals):
        configs.append(config_ext.get({'x': x}, resource=r))
    kernel = CrossValidationKernelFunction(
        kernel_main=ConstantKernelFunction(),
        kernel_residual=ConstantKernelFunction(),
        mean_main=ZeroMeanFunction(),
        num_folds=num_folds)
    inputs = config_ext.hp_ranges_ext.to_ndarray_matrix(configs)
    kern_mat = kernel(inputs, inputs)
    max_resources = np.maximum(rvals.reshape((-1, 1)), rvals.reshape((1, -1)))
    kern_mat_compare = (1.0 / max_resources) + 1.0
    np.testing.assert_almost_equal(kern_mat, kern_mat_compare)
