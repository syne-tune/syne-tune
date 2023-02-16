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
from typing import Optional, Tuple, List, Dict, Any
import autograd.numpy as anp

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import (
    DEFAULT_ENCODING,
    INITIAL_WARPING,
    WARPING_LOWER_BOUND,
    WARPING_UPPER_BOUND,
    NUMERICAL_JITTER,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.distribution import (
    LogNormal,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel import (
    KernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    MeanFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon_blocks_helpers import (
    encode_unwrap_parameter,
    register_parameter,
    create_encoding,
)
from syne_tune.util import is_positive_integer
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges import (
    HyperparameterRanges,
)
from syne_tune.config_space import Categorical, Ordinal


class Warping(MeanFunction):
    r"""
    Warping transform on contiguous range of feature :math:`x`. Each warped
    coordinate has two independent warping parameters.

    If :math:`x = [x_1, \dots, x_d]` and ``coordinate_range = (l, r)``, the
    warping transform operates on :math:`[x_l, \dots, x_{r-1}]`. The default
    for ``coordinate_range`` is the full range, and we must have ``l < r``.
    The block is the identity on all remaining coordinates. Input coordinates
    are assumed to lie in :math:`[0, 1]`. The warping transform on each
    coordinate is due to Kumaraswamy:

    .. math::

       warp(x_j) = 1 - (1 - r(x_j)^{a_j})^{b_j}.

    Here, :math:`r(x_j)` linearly maps :math:`[0, 1]` to
    :math:`[\epsilon, 1 - \epsilon]` for a small :math:`\epsilon > 0`, which
    avoids numerical issues when taking derivatives.

    :param dimension: Dimension :math:`d` of input
    :param coordinate_range: Range ``(l, r)``, see above. Default is
        ``(0, dimension)``, so the full range
    :param encoding_type: Encoding type
    """

    def __init__(
        self,
        dimension: int,
        coordinate_range: Optional[Tuple[int, int]] = None,
        encoding_type: str = DEFAULT_ENCODING,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert is_positive_integer([dimension])
        self.dimension = dimension
        if coordinate_range is None:
            coordinate_range = (0, dimension)
        else:
            l, r = coordinate_range
            assert (
                0 <= l < r <= dimension
            ), f"{coordinate_range} must be included in (0, {dimension})"
        self.lower, self.upper = coordinate_range
        size = self.upper - self.lower
        self._is_full_range = size == dimension
        self._rescale_mult = 1.0 - 2 * NUMERICAL_JITTER
        self._rescale_offset = NUMERICAL_JITTER
        self.encoding = create_encoding(
            encoding_name=encoding_type,
            init_val=INITIAL_WARPING,
            constr_lower=WARPING_LOWER_BOUND,
            constr_upper=WARPING_UPPER_BOUND,
            dimension=size,
            prior=LogNormal(0.0, 0.75),
        )
        with self.name_scope():
            self.power_a_internal = register_parameter(
                self.params, "power_a", self.encoding, shape=(size,)
            )
            self.power_b_internal = register_parameter(
                self.params, "power_b", self.encoding, shape=(size,)
            )

    def _rescale(self, x):
        return self._rescale_mult * x + self._rescale_offset

    def _warping(self):
        return (
            anp.reshape(
                encode_unwrap_parameter(self.power_a_internal, self.encoding), (1, -1)
            ),
            anp.reshape(
                encode_unwrap_parameter(self.power_b_internal, self.encoding), (1, -1)
            ),
        )

    def forward(self, x):
        """
        Actual computation of the warping transformation (see details above)

        :param x: Input data, shape ``(n, d)``
        """
        power_a, power_b = self._warping()
        if not self._is_full_range:
            x_part = x[:, self.lower : self.upper]
        else:
            x_part = x
        result = 1.0 - anp.power(
            1.0 - anp.power(self._rescale(x_part), power_a), power_b
        )
        if not self._is_full_range:
            args = []
            if self.lower > 0:
                args.append(x[:, : self.lower])
            args.append(result)
            if self.upper < self.dimension:
                args.append(x[:, self.upper :])
            result = anp.concatenate(args, axis=1)
        return result

    def param_encoding_pairs(self):
        return [
            (self.power_a_internal, self.encoding),
            (self.power_b_internal, self.encoding),
        ]

    @staticmethod
    def _param_name(kind: str, index: int, just_one: bool) -> str:
        if just_one:
            return "power_" + kind
        else:
            return f"power_{kind}_{index}"

    def get_params(self) -> Dict[str, Any]:
        size = self.upper - self.lower
        just_one = size == 1
        param_dict = dict()
        for kind, warping in zip(("a", "b"), self._warping()):
            warping = anp.reshape(warping, (-1,))
            param_dict.update(
                {
                    self._param_name(kind, index, just_one): warping[index]
                    for index in range(size)
                }
            )
        return param_dict

    def set_params(self, param_dict: Dict[str, Any]):
        size = self.upper - self.lower
        just_one = size == 1
        for kind in ("a", "b"):
            warping = [
                param_dict[self._param_name(kind, index, just_one)]
                for index in range(size)
            ]
            warping_int = (
                self.power_a_internal if kind == "a" else self.power_b_internal
            )
            self.encoding.set(warping_int, warping)


def warpings_for_hyperparameters(hp_ranges: HyperparameterRanges) -> List[Warping]:
    """
    It is custom to warp hyperparameters which are not categorical. This
    function creates warpings based on your configuration space.

    :param hp_ranges: Encoding of configuration space
    :return: To be used as ``warpings`` in :class:`WarpedKernel`
    """

    dimension = hp_ranges.ndarray_size
    lower = None
    dims = 0
    warpings = []
    for name in hp_ranges.internal_keys:
        hp_range = hp_ranges.config_space[name]
        is_choice = isinstance(hp_range, Categorical) and not isinstance(
            hp_range, Ordinal
        )
        if not is_choice:
            if lower is None:
                lower = dims
            dims += 1
        else:
            if lower is not None:
                coordinate_range = (lower, dims)
                warpings.append(Warping(dimension, coordinate_range))
                lower = None
            # For binary, we use a single dimension, not 2
            sz = len(hp_range.categories)
            if sz == 2:
                sz = 1
            dims += sz
    assert (
        dims == dimension
    ), f"Internal error: dimension = {dimension}, dims = {dims}, hp_ranges = {hp_ranges}"
    if lower is not None:
        coordinate_range = (lower, dims)
        warpings.append(Warping(dimension, coordinate_range))
    return warpings


def kernel_with_warping(
    kernel: KernelFunction, hp_ranges: HyperparameterRanges
) -> KernelFunction:
    """
    Note that the coordinates corresponding to categorical parameters are not
    warped.

    :param kernel: Kernel :math:`k(x, x')` without warping
    :param hp_ranges: Encoding of configuration space
    :return: Kernel with warping
    """
    warpings = warpings_for_hyperparameters(hp_ranges)
    if warpings:
        kernel = WarpedKernel(kernel=kernel, warpings=warpings)
    return kernel


class WarpedKernel(KernelFunction):
    """
    Block that composes warping with an arbitrary kernel. We allow for a
    list of warping transforms, so that a non-contiguous set of input
    coordinates can be warped.

    It is custom to warp hyperparameters which are not categorical. You can
    use :func:`kernel_with_warping` to furnish a kernel with warping for all
    non-categorical hyperparameters.

    :param kernel: Kernel :math:`k(x, x')`
    :param warpings: List of warping transforms, which are applied sequentially.
        Ranges of different entries should be non-overlapping, this is not
        checked.
    """

    def __init__(self, kernel: KernelFunction, warpings: List[Warping], **kwargs):
        super().__init__(kernel.dimension, **kwargs)
        num_warpings = len(warpings)
        assert num_warpings > 0
        assert all(
            kernel.dimension == warping.dimension for warping in warpings
        ), f"Dimensions of all entries in warping must be kernel.dimension = {kernel.dimension}"
        self.kernel = kernel
        self.warpings = warpings.copy()
        # Note: Child blocks in lists or dicts are not registered automatically
        for v in self.warpings:
            self.register_child(v)
        self._prefixes = ["kernel_"]
        if num_warpings == 1:
            self._prefixes.append("warping_")
        else:
            self._prefixes.extend(f"warping{k}_" for k in range(num_warpings))

    def _apply_warpings(self, X):
        warped_X = X
        for warping in self.warpings:
            warped_X = warping(warped_X)
        return warped_X

    def forward(self, X1, X2):
        warped_X1 = self._apply_warpings(X1)
        if X2 is X1:
            warped_X2 = warped_X1
        else:
            warped_X2 = self._apply_warpings(X2)
        return self.kernel(warped_X1, warped_X2)

    def diagonal(self, X):
        # If kernel.diagonal does not depend on content of X (but just its
        # size), can pass X instead of self.warping(X)
        warped_X = self._apply_warpings(X) if self.kernel.diagonal_depends_on_X() else X
        return self.kernel.diagonal(warped_X)

    def diagonal_depends_on_X(self):
        return self.kernel.diagonal_depends_on_X()

    def param_encoding_pairs(self):
        return self.kernel.param_encoding_pairs() + [
            x for warping in self.warpings for x in warping.param_encoding_pairs()
        ]

    def get_params(self) -> Dict[str, Any]:
        result = dict()
        blocks = [self.kernel] + self.warpings
        for pref, block in zip(self._prefixes, blocks):
            result.update({(pref + k): v for k, v in block.get_params().items()})
        return result

    def set_params(self, param_dict: Dict[str, Any]):
        blocks = [self.kernel] + self.warpings
        for pref, block in zip(self._prefixes, blocks):
            len_pref = len(pref)
            stripped_dict = {
                k[len_pref:]: v for k, v in param_dict.items() if k.startswith(pref)
            }
            block.set_params(stripped_dict)
