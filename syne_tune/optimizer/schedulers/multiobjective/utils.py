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

from syne_tune.try_import import try_import_moo_message

try:
    from pymoo.indicators.hv import HV
except ImportError:
    print(try_import_moo_message())


EPSILON = 1e-6


def hypervolume(
    results_array: np.ndarray,
    reference_point: np.ndarray = None,
) -> float:
    """
    Compute the hypervolume of all results based on reference points

    :param results_df: Array with experiment results ordered by time with shape (npoints, ndimensions)
    :param reference_point: Reference points for hypervolume calculations.
        If None, the maximum values of each dimension of results_array is used.
    """
    if reference_point is None:
        reference_point = results_array.max(axis=0) * (1 + EPSILON) + EPSILON
    indicator_fn = HV(ref_point=reference_point)
    return indicator_fn(results_array)


def hypervolume_cumulative(
    results_array: np.ndarray,
    reference_point: np.ndarray = None,
) -> np.ndarray:
    """
    Compute the cumulative hypervolume of all results based on reference points
    Returns an array with hypervolumes given by an increasing range of points.
    ``return_array[idx] = hypervolume(results_array[0: idx])``

    :param results_array: Array with experiment results ordered by time with shape (npoints, ndimensions).
    :param reference_point: Reference points for hypervolume calculations.
        If None, the maximum values of each dimension of results_array is used.
    :return: Cumulative hypervolume array with dimensions (npoints, )
    """
    if reference_point is None:
        reference_point = results_array.max(axis=0) * (1 + EPSILON) + EPSILON
    indicator_fn = HV(ref_point=reference_point)
    hypervolume_indicator = np.zeros(shape=len(results_array))
    for idx in range(len(results_array)):
        hypervolume_indicator[idx] = indicator_fn(results_array[0:idx])
    return hypervolume_indicator
