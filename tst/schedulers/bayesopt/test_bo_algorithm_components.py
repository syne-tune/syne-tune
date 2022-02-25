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
import pytest

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory \
    import make_hyperparameter_ranges
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import \
    dictionarize_objective
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import \
    TuningJobState
from syne_tune.config_space import uniform, choice, randint
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.test_objects \
    import create_tuning_job_state


@pytest.fixture(scope='function')
def tuning_job_state():
    hp_ranges1 = make_hyperparameter_ranges({
        'a1_hp_1': uniform(-5.0, 5.0),
        'a1_hp_2': choice(['a', 'b', 'c'])})
    X1 = [(-3.0, 'a'), (-1.9, 'c'), (-3.5, 'a')]
    Y1 = [dictionarize_objective(y) for y in (1.0, 2.0, 0.3)]
    hp_ranges2 = make_hyperparameter_ranges({
        'a1_hp_1': uniform(-5.0, 5.0),
        'a1_hp_2': randint(-5, 5)})
    X2 = [(-1.9, -1), (-3.5, 3)]
    Y2 = [dictionarize_objective(y) for y in (0.0, 2.0)]
    return {
        'algo-1': create_tuning_job_state(
            hp_ranges=hp_ranges1, cand_tuples=X1, metrics=Y1),
        'algo-2': create_tuning_job_state(
            hp_ranges=hp_ranges2, cand_tuples=X2, metrics=Y2),
    }


@pytest.fixture(scope='function')
def tuning_job_sub_state():
    hp_ranges = make_hyperparameter_ranges(dict())
    return TuningJobState.empty_state(hp_ranges)
