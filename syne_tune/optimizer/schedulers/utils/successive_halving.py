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
from typing import Optional, List
import logging
import numpy as np

logger = logging.getLogger(__name__)


def _is_positive_int(x):
    return int(x) == x and x >= 1


def successive_halving_rung_levels(
    rung_levels: Optional[List[int]],
    grace_period: int,
    reduction_factor: Optional[float],
    rung_increment: Optional[int],
    max_t: int,
) -> List[int]:
    """Creates ``rung_levels`` from ``grace_period``, ``reduction_factor``

    Note: If ``rung_levels`` is given and ``rung_levels[-1] == max_t``, we strip
    off this final entry, so that all rung levels are ``< max_t``.

    :param rung_levels: If given, this is returned (but see above)
    :param grace_period: See :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`
    :param reduction_factor: See :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`
    :param rung_increment: See :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`
    :param max_t: See :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`
    :return: List of rung levels
    """
    if rung_levels is not None:
        assert (
            isinstance(rung_levels, list) and len(rung_levels) > 1
        ), "rung_levels must be list of size >= 2"
        assert all(
            _is_positive_int(x) for x in rung_levels
        ), "rung_levels must be list of positive integers"
        rung_levels = [int(x) for x in rung_levels]
        assert all(
            x < y for x, y in zip(rung_levels, rung_levels[1:])
        ), "rung_levels must be strictly increasing sequence"
        assert (
            rung_levels[-1] <= max_t
        ), f"Last entry of rung_levels ({rung_levels[-1]}) must be <= max_t ({max_t})"
    else:
        # Rung levels given by grace_period, reduction_factor, max_t
        assert _is_positive_int(grace_period)
        assert _is_positive_int(max_t)
        assert (
            max_t > grace_period
        ), f"max_t ({max_t}) must be greater than grace_period ({grace_period})"
        if reduction_factor is not None:
            assert reduction_factor >= 2
            rf = reduction_factor
            min_t = grace_period
            max_rungs = 0
            while min_t * np.power(rf, max_rungs) < max_t:
                max_rungs += 1
            rung_levels = [
                int(round(min_t * np.power(rf, k))) for k in range(max_rungs)
            ]
            assert rung_levels[-1] <= max_t  # Sanity check
            if rung_increment is not None:
                logger.warning(
                    f"You specified both reduction_factor = {reduction_factor} "
                    f"and rung_increment = {rung_increment}. The former takes "
                    "precedence, the latter will be ignored"
                )
        else:
            assert rung_increment is not None
            assert _is_positive_int(rung_increment)
            rung_levels = list(range(grace_period, max_t, rung_increment))
    if rung_levels[-1] == max_t:
        rung_levels = rung_levels[:-1]
    return rung_levels
