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
import numpy as np

from syne_tune.config_space import uniform, randint, choice
from syne_tune.optimizer.schedulers.searchers.gp_searcher_factory import \
    gp_fifo_searcher_defaults, gp_fifo_searcher_factory
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments \
    import check_and_merge_defaults
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel \
    import Matern52, ProductKernelFunction


def test_create_transfer_learning():
    config_space = {
        'task_id': choice(['0', '1', '2', '3']),
        'a': uniform(lower=0.0, upper=2.0),
        'b': randint(lower=2, upper=5),
        'c': choice(['a', 'b', 'c']),
    }
    active_config_space = {
        'a': uniform(lower=0.2, upper=0.8),
        'b': randint(lower=2, upper=4),
        'c': choice(['a', 'c']),
    }
    search_options = {
        'scheduler': 'fifo',
        'config_space': config_space,
        'transfer_learning_task_attr': 'task_id',
        'transfer_learning_active_task': '2',
        'transfer_learning_active_config_space': active_config_space,
    }
    kwargs = check_and_merge_defaults(
        search_options, *gp_fifo_searcher_defaults(),
        dict_name='search_options')
    kwargs_int = gp_fifo_searcher_factory(**kwargs)

    filter_observed_data = kwargs_int.get('filter_observed_data')
    assert filter_observed_data is not None
    config = dict(task_id='2', a=0.0, b=2, c='a')
    assert filter_observed_data(config)
    config = dict(task_id='0', a=0.0, b=2, c='a')
    assert not filter_observed_data(config)

    hp_ranges = kwargs_int['hp_ranges']
    assert hp_ranges.internal_keys == ['task_id', 'a', 'b', 'c']
    assert hp_ranges.ndarray_size() == 9
    expected_ndarray_bounds = [
        (0.0, 0.0), (0.0, 0.0), (1.0, 1.0), (0.0, 0.0),
        (0.1, 0.4),
        (0.0, 0.75),
        (0.0, 1.0), (0.0, 0.0), (0.0, 1.0)]
    ndarray_bounds = hp_ranges.get_ndarray_bounds()
    mat = np.array([[x[i] for x in ndarray_bounds] for i in range(2)])
    expected_mat = np.array([[x[i] for x in expected_ndarray_bounds]
                             for i in range(2)])
    np.testing.assert_almost_equal(mat, expected_mat)

    kernel = kwargs_int['model_factory']._gpmodel.likelihood.kernel
    assert isinstance(kernel, ProductKernelFunction)
    kernel1 = kernel.kernel1
    assert isinstance(kernel1, Matern52)
    assert kernel1.dimension == 4
    assert not kernel1.ARD
    kernel2 = kernel.kernel2
    assert isinstance(kernel2, Matern52)
    assert kernel2.dimension == 5
    assert kernel2.ARD
