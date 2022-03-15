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

from syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model \
    import get_internal_candidate_evaluations
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import dictionarize_objective, \
    INTERNAL_METRIC_NAME
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.test_objects \
    import dimensionality_and_warping_ranges, create_tuning_job_state
from syne_tune.config_space import uniform, randint, choice, loguniform
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory \
    import make_hyperparameter_ranges


def test_get_internal_candidate_evaluations():
    """we do not test the case with no evaluations, as it is assumed
    that there will be always some evaluations generated in the beginning
    of the BO loop."""

    hp_ranges = make_hyperparameter_ranges({
        'a': randint(0, 10),
        'b': uniform(0.0, 10.0),
        'c': choice(['X', 'Y'])})
    cand_tuples = [
        (2, 3.3, 'X'),
        (1, 9.9, 'Y'),
        (7, 6.1, 'X')]
    metrics = [dictionarize_objective(y) for y in (5.3, 10.9, 13.1)]

    state = create_tuning_job_state(
        hp_ranges=hp_ranges, cand_tuples=cand_tuples, metrics=metrics)
    state.failed_trials.append('0')  # First trial with observation also failed

    result = get_internal_candidate_evaluations(
        state, INTERNAL_METRIC_NAME, normalize_targets=True,
        num_fantasy_samples=20)

    assert len(result.features.shape) == 2, "Input should be a matrix"
    assert len(result.targets.shape) == 2, "Output should be a matrix"

    assert result.features.shape[0] == len(cand_tuples)
    assert result.targets.shape[-1] == 1, \
        "Only single output value per row is suppored"

    assert np.abs(np.mean(result.targets)) < 1e-8, \
        "Mean of the normalized outputs is not 0.0"
    assert np.abs(np.std(result.targets) - 1.0) < 1e-8, \
        "Std. of the normalized outputs is not 1.0"

    np.testing.assert_almost_equal(result.mean, 9.766666666666666)
    np.testing.assert_almost_equal(result.std, 3.283629428273267)


def test_dimensionality_and_warping_ranges():
    # Note: `choice` with binary value range is encoded as 1, not 2 dims
    hp_ranges = make_hyperparameter_ranges({
        'a': choice(['X', 'Y']),  # pos 0
        'b': loguniform(0.1, 10.0),  # pos 1
        'c': choice(['a', 'b', 'c']),  # pos 2
        'd': uniform(0.0, 10.0),  # pos 5
        'e': choice(['X', 'Y'])})  # pos 6

    dim, warping_ranges = dimensionality_and_warping_ranges(hp_ranges)
    assert dim == 7
    assert warping_ranges == {1: (0.0, 1.0), 5: (0.0, 1.0)}
