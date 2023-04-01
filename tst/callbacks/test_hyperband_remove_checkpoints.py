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
from typing import List, Tuple, Dict, Any
import numpy as np
from scipy.special import betainc

from syne_tune.callbacks.hyperband_remove_checkpoints_callback import (
    HyperbandRemoveCheckpointsCallback,
)
from syne_tune.callbacks.hyperband_remove_checkpoints_score import (
    compute_probabilities_of_getting_resumed,
)


def _q_vals_betainc(
    l_val: int, p_val: float
) -> Tuple[float, float, float, float, float]:
    u_vals = np.arange(5)
    cdf_vals = betainc(
        np.maximum(l_val - u_vals, 1e-7), u_vals + 1, np.array([1 - p_val])
    )
    cdf_vals[u_vals >= l_val] = 1
    c0, c1, c2, c3, c4 = cdf_vals
    return c0, c1 - c0, c2 - c1, c3 - c2, c4 - c3


def _q_vals(
    n_val: int, p_val: float, s_pos: int, i_pos: int, extra: Dict[str, Any]
) -> Tuple[float, float, float, float, float]:
    r"""
    Probabilities of binomial :math:`\mathrm{bin}(l, p)` to be equal to
    ``0, 1, 2, 3, 4``.
    """
    assert n_val >= 1
    p_ratio = p_val / (1 - p_val)
    q0 = np.power(1 - p_val, n_val)
    q1 = n_val * p_ratio * q0
    q2 = ((n_val - 1) * p_ratio * q1 / 2) * (n_val >= 2)
    q3 = ((n_val - 2) * p_ratio * q2 / 3) * (n_val >= 3)
    q4 = ((n_val - 3) * p_ratio * q3 / 4) * (n_val >= 4)
    np.testing.assert_almost_equal(
        _q_vals_betainc(n_val, p_val),
        [q0, q1, q2, q3, q4],
        decimal=5,
    )
    extra["q_vals"][i_pos, :, s_pos] = np.array([q4, q3, q2, q1, q0])
    return q0, q1, q2, q3, q4


def _append_column(m_vals: List[np.ndarray], i: int, col: np.ndarray):
    num_m = len(m_vals)
    col = col.reshape((-1, 1))
    if i >= num_m:
        assert i == num_m
        m_vals.append(col)
    else:
        m_vals[i] = np.hstack((m_vals[i], col))


def compute_probability_resumed(
    l_ind: List[int], p_val: float, s_pos: int, extra: Dict[str, Any]
) -> float:
    """
    Here, ``l_ind`` must have length 5
    """
    assert len(l_ind) == 5
    l0, l1, l2, l3, l4 = l_ind
    m00, m01, m02, m03, m04 = _q_vals(l0, p_val, s_pos, 0, extra)
    q10, q11, q12, q13, _ = _q_vals(l1 - l0, p_val, s_pos, 1, extra)
    q20, q21, q22, _, _ = _q_vals(l2 - l1, p_val, s_pos, 2, extra)
    q30, q31, _, _, _ = _q_vals(l3 - l2, p_val, s_pos, 3, extra)
    q40, _, _, _, _ = _q_vals(l4 - l3, p_val, s_pos, 3, extra)
    m11 = m01 * q10
    m12 = m01 * q11 + m02 * q10
    m22 = m12 * q20
    m13 = m01 * q12 + m02 * q11 + m03 * q10
    m23 = m12 * q21 + m13 * q20
    m33 = m23 * q30
    m14 = m01 * q13 + m02 * q12 + m03 * q11 + m04 * q10
    m24 = m12 * q22 + m13 * q21 + m14 * q20
    m34 = m23 * q31 + m24 * q30
    m44 = m34 * q40
    if "m_vals" not in extra:
        extra["m_vals"] = []
    m_vals = extra["m_vals"]
    _append_column(m_vals, i=0, col=np.array([m01, m02, m03, m04]))
    _append_column(m_vals, i=1, col=np.array([m12, m13, m14]))
    _append_column(m_vals, i=2, col=np.array([m23, m24]))
    _append_column(m_vals, i=3, col=np.array([m34]))
    return m00 + m11 + m22 + m33 + m44


def test_compute_probabilities_of_getting_resumed():
    random_seed = 31415938
    random_state = np.random.RandomState(random_seed)
    num_probs = 100
    approx_steps = 4

    prom_quants = random_state.uniform(low=0.25, high=0.5, size=num_probs)
    p_vals = random_state.uniform(low=0.1, high=0.9, size=num_probs)
    rung_lens = random_state.randint(low=10, high=100, size=num_probs)
    mult = random_state.uniform(low=1.1, high=1.5, size=num_probs)
    ranks = np.ceil(mult * rung_lens * prom_quants)
    c_vals = ranks - prom_quants * rung_lens
    l_indices = np.ceil(
        (c_vals + np.arange(0, approx_steps + 1).reshape((-1, 1))) / prom_quants
    )
    assert l_indices.shape == (approx_steps + 1, num_probs)
    # Choose large enough that nothing is thresholded
    l_max = l_indices[-1]
    time_ratio = 1.1 * np.max(l_max / rung_lens)
    target_probs = compute_probabilities_of_getting_resumed(
        ranks=ranks,
        rung_lens=rung_lens,
        prom_quants=prom_quants,
        p_vals=p_vals,
        time_ratio=time_ratio,
        approx_steps=approx_steps,
    )
    rs = approx_steps + 1
    extra = {"q_vals": np.zeros((rs, rs, num_probs))}
    target_props_compare = np.array(
        [
            compute_probability_resumed(
                l_ind=list(l_indices[:, s_pos]),
                p_val=p_vals[s_pos],
                s_pos=s_pos,
                extra=extra,
            )
            for s_pos in range(num_probs)
        ]
    )
    # DEBUG
    # q_vals = extra["q_vals"]
    # for s in range(q_vals.shape[2]):
    #     print(f"[compare] q_vals[{s}]:\n{q_vals[:, :, s]}")
    # m_vals = extra["m_vals"]
    # for i, mmat in enumerate(m_vals):
    #     print(f"[compare] m_vals[{i}]:\n{mmat}")
    # END DEBUG
    np.testing.assert_almost_equal(target_probs, target_props_compare, decimal=5)
