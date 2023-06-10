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
from typing import List
import numpy as np

from syne_tune.try_import import try_import_moo_message

try:
    from pymoo.indicators.hv import HV
except ImportError:
    print(try_import_moo_message())


EPSILON = 1e-6


def default_reference_point(results_array: np.ndarray) -> np.ndarray:
    return results_array.max(axis=0) * (1 + EPSILON) + EPSILON


def hypervolume(
    results_array: np.ndarray,
    reference_point: np.ndarray = None,
) -> float:
    """
    Compute the hypervolume of all results based on reference points

    :param results_array: Array with experiment results ordered by time with
        shape ``(npoints, ndimensions)``.
    :param reference_point: Reference points for hypervolume calculations.
        If ``None``, the maximum values of each dimension of results_array is
        used.
    :return Hypervolume indicator
    """
    if reference_point is None:
        reference_point = default_reference_point(results_array)
    indicator_fn = HV(ref_point=reference_point)
    return indicator_fn(results_array)


def linear_interpolate(hv_indicator: np.ndarray, indices: List[int]):
    for first, last in zip(indices[:-1], indices[1:]):
        num = last - first + 1
        v_first = hv_indicator[first]
        v_last = hv_indicator[last]
        hv_indicator[first : (last + 1)] = np.linspace(v_first, v_last, num=num)


# TODO: Use code for incremental hypervolume (adding one more point). Computation
# here can be slow.
# At least we could check for each new row if it is dominated by rows before. If
# so, the cumulative indicator remains the same.
def hypervolume_cumulative(
    results_array: np.ndarray,
    reference_point: np.ndarray = None,
    increment: int = 1,
) -> np.ndarray:
    """
    Compute the cumulative hypervolume of all results based on reference points
    Returns an array with hypervolumes given by an increasing range of points.
    ``return_array[idx] = hypervolume(results_array[0 : (idx + 1)])``.

    The current implementation is very slow, since the hypervolume index is not
    computed incrementally. A solution for now is to use ``increment > 1``,
    in which case the HV index is only computed every ``increment`` entry, and
    linearly interpolated in between.

    :param results_array: Array with experiment results ordered by time with
        shape ``(npoints, ndimensions)``.
    :param reference_point: Reference points for hypervolume calculations.
        If ``None``, the maximum values of each dimension of results_array is
        used.
    :return: Cumulative hypervolume array, shape ``(npoints,)``
    """
    if reference_point is None:
        reference_point = default_reference_point(results_array)
    indicator_fn = HV(ref_point=reference_point)
    hypervolume_indicator = np.zeros(shape=len(results_array))
    sz = len(results_array)
    indices = list(range(0, sz, increment))
    if indices[-1] != sz - 1:
        indices.append(sz - 1)
    for idx in indices:
        hypervolume_indicator[idx] = indicator_fn(results_array[0 : (idx + 1)])
    if increment > 1:
        linear_interpolate(hypervolume_indicator, indices)
    return hypervolume_indicator
