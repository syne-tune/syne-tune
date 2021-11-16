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
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel \
    import KernelFunction, Matern52, ExponentialDecayResourcesKernelFunction, \
    ExponentialDecayResourcesMeanFunction
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.warping \
    import WarpedKernel, Warping
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean \
    import MeanFunction


def resource_kernel_factory(
        name: str, kernel_x: KernelFunction, mean_x: MeanFunction) -> \
        (KernelFunction, MeanFunction):
    """
    Given kernel function kernel_x and mean function mean_x over config x,
    create kernel and mean functions over (x, r), where r is the resource
    attribute (nonnegative scalar, usually in [0, 1]).

    :param name: Selects resource kernel type
    :param kernel_x: Kernel function over configs x
    :param mean_x: Mean function over configs x
    :return: res_kernel, res_mean, both over (x, r)

    """
    if name == 'matern52':
        res_kernel = Matern52(dimension=kernel_x.dimension + 1, ARD=True)
        res_mean = mean_x
    elif name == 'matern52-res-warp':
        # Warping on resource dimension (last one)
        dim_x = kernel_x.dimension
        res_warping = Warping(
            dimension=dim_x + 1, index_to_range={dim_x: (0., 1.)})
        res_kernel = WarpedKernel(
            kernel=Matern52(dimension=dim_x + 1, ARD=True),
            warping=res_warping)
        res_mean = mean_x
    else:
        if name == 'exp-decay-sum':
            delta_fixed_value = 0.0
        elif name == 'exp-decay-combined':
            delta_fixed_value = None
        elif name == 'exp-decay-delta1':
            delta_fixed_value = 1.0
        else:
            raise AssertionError("name = '{}' not supported".format(name))
        res_kernel = ExponentialDecayResourcesKernelFunction(
            kernel_x, mean_x, delta_fixed_value=delta_fixed_value)
        res_mean = ExponentialDecayResourcesMeanFunction(
            kernel=res_kernel)

    return res_kernel, res_mean
