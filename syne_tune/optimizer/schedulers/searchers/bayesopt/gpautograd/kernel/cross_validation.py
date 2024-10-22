import autograd.numpy as anp
from autograd.builtins import isinstance
from typing import Dict, Any

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel.base import (
    KernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    MeanFunction,
)


def decode_resource_values(res_encoded, num_folds):
    """
    We assume the resource attribute ``r`` is encoded as
    :code:`randint(1, num_folds)`. Internally, ``r`` is taken as value in the
    real interval ``[0.5, num_folds + 0.5]``, which is linearly transformed to
    ``[0, 1]`` for encoding.

    :param res_encoded: Encoded values ``r``
    :param num_folds: Maximum number of folds
    :return: Original values ``r`` (not rounded to ``int``)
    """
    return res_encoded * num_folds + 0.5


class CrossValidationKernelFunction(KernelFunction):
    r"""
    Kernel function suitable for :math:`f(x, r)` being the average of ``r``
    validation metrics evaluated on different (train, validation) splits.

    More specifically, there are 'num_folds`` such splits, and :math:`f(x, r)`
    is the average over the first ``r`` of them.

    We model the score on fold ``k`` as :math:`e_k(x) = f(x) + g_k(x)`,
    where :math:`f(x)` and the :math:`g_k(x)` are a priori independent Gaussian
    processes with kernels ``kernel_main`` and ``kernel_residual`` (all :math:`g_k`
    share the same kernel). Moreover, the :math:`g_k` are zero-mean, while
    :math:`f(x)` may have a mean function. Then:

    .. math::

       f(x, r) = r^{-1} sum_{k \le r} e_k(x),

       k((x, r), (x', r')) = k_{main}(x, x') +
          \frac{k_{residual}(x, x')}{\mathrm{max}(r, r')}

    Note that ``kernel_main``, ``kernel_residual`` are over inputs :math:`x`
    (dimension ``d``), while the kernel represented here is over inputs
    :math:`(x, r)` of dimension ``d + 1``, where the resource attribute :math:`r`
    (number of folds) is last.

    Inputs are encoded. We assume a linear encoding for r with bounds 1 and
    ``num_folds``.
    TODO: Right now, all HPs are encoded, and the resource attribute counts as
    HP, even if it is not optimized over. This creates a dependence to how
    inputs are encoded.
    """

    def __init__(
        self,
        kernel_main: KernelFunction,
        kernel_residual: KernelFunction,
        mean_main: MeanFunction,
        num_folds: int,
        **kwargs,
    ):
        """
        :param kernel_main: Kernel for main effect :math:`f(x)`
        :param kernel_residual: Kernel for residuals :math:`g_k(x)`
        :param mean_main: Mean function for main effect :math:`f(x)`
        :param num_folds: Maximum number of folds: ``1 <= r <= num_folds``
        """
        super().__init__(dimension=kernel_main.dimension + 1, **kwargs)
        assert kernel_main.dimension == kernel_residual.dimension, (
            f"kernel_main.dimension = {kernel_main.dimension} != "
            + f"{kernel_residual.dimension} = kernel_residual.dimension"
        )
        assert (
            round(num_folds) == num_folds and num_folds >= 2
        ), f"num_folds = {num_folds} must be int >= 2"
        self.kernel_main = kernel_main
        self.kernel_residual = kernel_residual
        self.mean_main = mean_main
        self.num_folds = num_folds

    def _compute_terms(self, X):
        dim = self.kernel_main.dimension
        cfg = X[:, :dim]
        res_encoded = X[:, dim:]
        res_decoded = decode_resource_values(res_encoded, self.num_folds)
        return cfg, res_decoded

    def forward(self, X1, X2, **kwargs):
        cfg1, res1 = self._compute_terms(X1)
        if X2 is not X1:
            cfg2, res2 = self._compute_terms(X2)
        else:
            cfg2, res2 = cfg1, res1
        res1 = anp.reshape(res1, (-1, 1))
        res2 = anp.reshape(res2, (1, -1))

        kmat_main = self.kernel_main(cfg1, cfg2)
        kmat_residual = self.kernel_residual(cfg1, cfg2)
        max_resources = anp.maximum(res1, res2)

        return (kmat_residual / max_resources) + kmat_main

    def diagonal(self, X):
        cfg, res = self._compute_terms(X)
        res = anp.reshape(res, (-1,))
        kdiag_main = self.kernel_main.diagonal(cfg)
        kdiag_residual = self.kernel_residual.diagonal(cfg)

        return (kdiag_residual / res) + kdiag_main

    def diagonal_depends_on_X(self):
        return True

    def param_encoding_pairs(self):
        enc_list = []
        enc_list.extend(self.kernel_main.param_encoding_pairs())
        enc_list.extend(self.kernel_residual.param_encoding_pairs())
        enc_list.extend(self.mean_main.param_encoding_pairs())
        return enc_list

    def mean_function(self, X):
        cfg, _ = self._compute_terms(X)
        return self.mean_main(cfg)

    def get_params(self) -> Dict[str, Any]:
        result = dict()
        for pref, func in [
            ("kernelm_", self.kernel_main),
            ("meanm_", self.mean_main),
            ("kernelr_", self.kernel_residual),
        ]:
            result.update({(pref + k): v for k, v in func.get_params().items()})
        return result

    def set_params(self, param_dict: Dict[str, Any]):
        for pref, func in [
            ("kernelm_", self.kernel_main),
            ("meanm_", self.mean_main),
            ("kernelr_", self.kernel_residual),
        ]:
            len_pref = len(pref)
            stripped_dict = {
                k[len_pref:]: v for k, v in param_dict.items() if k.startswith(pref)
            }
            func.set_params(stripped_dict)


class CrossValidationMeanFunction(MeanFunction):
    def __init__(self, kernel: CrossValidationKernelFunction, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(kernel, CrossValidationKernelFunction)
        self.kernel = kernel

    def forward(self, X):
        return self.kernel.mean_function(X)

    def param_encoding_pairs(self):
        return []

    def get_params(self) -> Dict[str, Any]:
        return dict()

    def set_params(self, param_dict: Dict[str, Any]):
        pass
