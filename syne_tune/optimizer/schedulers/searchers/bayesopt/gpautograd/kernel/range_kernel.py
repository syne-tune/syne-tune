from typing import Dict, Any
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel.base import (
    KernelFunction,
)


class RangeKernelFunction(KernelFunction):
    r"""
    Given kernel function ``K`` and range ``R``, this class represents

    .. math::

       (x, y) \mapsto K(x_R, y_R)

    """

    def __init__(self, dimension: int, kernel: KernelFunction, start: int, **kwargs):
        """
        :param dimension: Input dimension
        :param kernel: Kernel function K
        :param start: Range is ``range(start, start + kernel.dimension)``

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

    def get_params(self) -> Dict[str, Any]:
        return self.kernel.get_params()

    def set_params(self, param_dict: Dict[str, Any]):
        self.kernel.set_params(param_dict)
