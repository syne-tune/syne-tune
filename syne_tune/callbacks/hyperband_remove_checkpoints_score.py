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
import logging

from syne_tune.try_import import try_import_gpsearchers_message

try:
    from scipy.special import betainc
except ImportError:
    logging.info(try_import_gpsearchers_message())


def _binomial_cdf(
    u_vals: np.ndarray, n_vals: np.ndarray, p_vals: np.ndarray
) -> np.ndarray:
    r"""
    Computes binomial cumulative distribution function :math:`P(X \le u)`, where
    :math:`X\sim \mathrm{bin}(n, p)`.

    :param u_vals: Values for :math:`u`
    :param n_vals: Values for :math:`n`
    :param p_vals: Values for :math:`p`
    :return: CDF values
    """
    a_vals = np.maximum(n_vals - u_vals, 1e-7)
    b_vals = np.maximum(u_vals + 1, 1e-7)
    result = betainc(a_vals, b_vals, 1 - p_vals)
    result[u_vals >= n_vals] = 1
    return result


def _compute_binomial_probabilities(
    l_indices: np.ndarray, p_vals: np.ndarray
) -> np.ndarray:
    r"""
    Computes the 3D tensor with entries

    .. math::
       q_{i, j}^{(s)}
       = P\left\{ \mathrm{bin}(l_i - l_{i-1}, p_r) = j \right\}, \\
       l_i = l_i^{(s)},\; p_r = p_{r_s},

    its dimension is ``(rs, rs, n_scores)``. Note that the position
    ``(i, j, s)`` maps to :math:`q_{i, r_* - j}^{(s)}`, so the ordering in the
    ``j`` index is inverted.

    We could save some time by noting that some ``(i, s)`` entries may be the
    same, but don't do this here.

    :param l_indices: :math:`[l_i^{(s)}]`, shape ``(rs, n_scores)``
    :param p_vals: Probabilities :math:`[p_{r_s}]`, shape ``(1, n_scores)``
    :return: See above, shape ``(rs, rs, n_scores)``
    """
    rs = l_indices.shape[0]
    l_diffs = np.vstack((l_indices[0], l_indices[1:, :] - l_indices[:-1, :]))
    # Tensor index is ``(i, j, s)``
    # Many of these CDF values could be = 1, and we could maybe save some
    # time here:
    cdf_vals = _binomial_cdf(
        u_vals=np.arange(rs - 1, -1, -1).reshape((1, -1, 1)),
        n_vals=l_diffs.reshape((rs, 1, -1)),
        p_vals=p_vals.reshape((1, 1, -1)),
    )
    # CDF values are :math:`P(X(l_i - l_{i-1}, p_r) <= r_* - j)`. We need
    # probabilities :math:`P(X(l_i - l_{i-1}, p_r) = r_* - j)`, which we
    # get by finite differences along the ``j`` dimension. Note the order of
    # the ``j`` index is reversed
    q_vals = np.concatenate(
        (
            cdf_vals[:, :-1, :] - cdf_vals[:, 1:, :],
            cdf_vals[:, -1, :].reshape((rs, 1, -1)),
        ),
        axis=1,
    )
    return q_vals


def compute_probabilities_of_getting_resumed(
    ranks: np.ndarray,
    rung_lens: np.ndarray,
    prom_quants: np.ndarray,
    p_vals: np.ndarray,
    time_ratio: float,
    approx_steps: int,
) -> np.ndarray:
    r"""
    Computes an approximation to the probability of getting resumed under our
    independence assumptions. This approximation improves with larger
    ``approx_steps``, but its cost scales cubically in this number.

    :param ranks: Ranks :math:`k`, starting from 1 (smaller is better)
    :param rung_lens: Rung lengths :math:`n_r`
    :param prom_quants: Promotion quantiles :math:`\alpha_r`
    :param p_vals: Probabilities :math:`p_r`
    :param time_ratio: Ratio :math:`\beta` between time left and time spent
    :param approx_steps: Number of approximation steps, see above
    :return: Approximations of probability to get resumed
    """
    assert time_ratio > 0
    approx_steps = int(approx_steps)
    assert approx_steps >= 1
    # Precompute :math:`l_r` and binomial probabilities
    c_vals = ranks - prom_quants * rung_lens
    orig_num_trials = c_vals.size
    cpos_indicator = c_vals > 0
    if not all(cpos_indicator):
        # For some candidates, the condition is already fulfilled. For these,
        # we return 1
        non_trivial_index = np.nonzero(cpos_indicator)[0]
        c_vals = c_vals[non_trivial_index]
        prom_quants = prom_quants[non_trivial_index]
        p_vals = p_vals[non_trivial_index]
        rung_lens = rung_lens[non_trivial_index]
    else:
        non_trivial_index = None
    num_trials = c_vals.size
    c_vals = c_vals.reshape((1, -1))
    prom_quants = prom_quants.reshape((1, -1))
    p_vals = p_vals.reshape((1, -1))
    # :math:`l_i = \lceil (C + i) / \alpha_r \rceil`
    l_indices = np.ceil(
        (c_vals + np.arange(0, approx_steps + 1).reshape((-1, 1))) / prom_quants
    )
    n_prime = np.ceil(rung_lens * time_ratio)  # Maximum lengths
    # We can stop once :math:`l_r > n'`. Maybe, this allows to reduce the overall
    # ``approx_steps``
    max_r = np.max(np.sum(l_indices <= n_prime.reshape((1, -1)), axis=0)) - 1
    approx_steps = max(min(approx_steps, max_r), 1)
    if approx_steps < l_indices.shape[0] - 1:
        l_indices = l_indices[: (approx_steps + 1), :]
    # Tensor :math:`q_{i,j}^{(s)}`. Note that index ``(i, j, s)`` maps to
    # :math:`q_{i,r_* - j}^{(s)}`, where :math:`r_*` is ``approx_steps``, so
    # the ordering is reversed on the ``j`` dimension
    q_vals = _compute_binomial_probabilities(l_indices, p_vals)

    # Dynamic programming: Compute :math:`M_i` probabilities and sum up
    # target probabilities
    m_vals = [np.zeros((0, num_trials)) for _ in range(approx_steps)]
    # Start with :math:``m_{0,0} = q_{0,0}``
    target_probs = q_vals[0, approx_steps, :].reshape((-1,))
    for r in range(1, approx_steps + 1):
        m_row = q_vals[0, approx_steps - r, :]  # :math:`m_{0,r} = q_{0,r}`
        m_vals[0] = np.vstack((m_vals[0], m_row))
        for i in range(1, r + 1):
            zero_indicator = l_indices[i] > n_prime
            if not all(zero_indicator):
                num_j = r - i + 1
                qtil_mat = q_vals[i, -num_j:, :].reshape((num_j, -1))
                assert qtil_mat.shape == m_vals[i - 1].shape, (
                    qtil_mat.shape,
                    m_vals[i - 1].shape,
                )
                m_row = np.sum(qtil_mat * m_vals[i - 1], axis=0)
                m_row[zero_indicator] = 0
            else:
                m_row = np.zeros(num_trials)
            # ``m_row`` is :math:`m_{i,r}`
            if i < r:
                m_vals[i] = np.vstack((m_vals[i], m_row))
            else:
                target_probs += m_row

    if non_trivial_index is not None:
        tempvec = target_probs
        target_probs = np.ones(orig_num_trials)
        target_probs[non_trivial_index] = tempvec
    return target_probs
