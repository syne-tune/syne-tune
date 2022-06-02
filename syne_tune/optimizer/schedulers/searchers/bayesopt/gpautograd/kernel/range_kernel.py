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
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel.base import (
    KernelFunction,
)

__all__ = ["RangeKernelFunction"]


class RangeKernelFunction(KernelFunction):
    """
    Given kernel function K and range R, this class represents

        (x, y) -> K(x[R], y[R])

    """

    def __init__(self, dimension: int, kernel: KernelFunction, start: int, **kwargs):
        """
        :param dimension: Input dimension
        :param kernel: Kernel function K
        :param start: Range is `range(start, start + kernel.dimension)`

        """
        super().__init__(dimension, **kwargs)
        assert start >= 0 and start + kernel.dimension <= dimension, (
            start,
            dimension,
            kernel.dimension,
        )
        self.kernel = kernel
        self.start = start

    def forward(self, X1, X2):
        a = self.start
        b = a + self.kernel.dimension
        X1_part = X1[:, a:b]
        if X2 is X1:
            X2_part = X1_part
        else:
            X2_part = X2[:, a:b]
        return self.kernel(X1_part, X2_part)

    def diagonal(self, X):
        a = self.start
        b = a + self.kernel.dimension
        return self.kernel.diagonal(X[:, a:b])

    def diagonal_depends_on_X(self):
        return self.kernel.diagonal_depends_on_X()

    def param_encoding_pairs(self):
        """
        Note: We assume that K1 and K2 have disjoint parameters, otherwise
        there will be a redundancy here.
        """
        return self.kernel.param_encoding_pairs()

    def get_params(self):
        return self.kernel.get_params()

    def set_params(self, param_dict):
        self.kernel.set_params(param_dict)
