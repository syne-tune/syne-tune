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

from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.duplicate_detector \
    import DuplicateDetectorIdentical, DuplicateDetectorNoDetection
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory \
    import make_hyperparameter_ranges
from syne_tune.config_space import uniform, randint, choice
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.test_objects \
    import create_exclusion_set


hp_ranges = make_hyperparameter_ranges({
    'hp1': randint(0, 1000000000),
    'hp2': uniform(-10.0, 10.0),
    'hp3': choice(['a', 'b', 'c'])})


@pytest.mark.parametrize('existing, new, contained', [
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (10000, 3.0, 'c'), False),
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (10, 1.000001, 'a'), False),
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (20, 2.000001, 'b'), False),
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (25, 1.0, 'a'), False),
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (10, 1.0, 'a'), True),
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (20, 2.0, 'b'), True),
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (19, 1.0, 'a'), False),
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (10, 1.0000001, 'a'), False),
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (10, 1.0, 'c'), False),
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (10, 1.0, 'b'), False),
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (20, 1.0, 'b'), False),
])
def test_contains_identical(existing, new, contained):
    existing = create_exclusion_set(existing, hp_ranges)
    new = hp_ranges.tuple_to_config(new)
    assert DuplicateDetectorIdentical().contains(existing, new) == contained


@pytest.mark.parametrize('existing, new', [
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (10000, 3.0, 'c')),
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (10, 1.000001, 'a')),
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (20, 2.000001, 'b')),
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (25, 1.0, 'a')),
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (10, 1.0, 'a')),
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (20, 2.0, 'b')),
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (19, 1.0, 'a')),
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (10, 1.0000001, 'a')),
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (10, 1.0, 'c')),
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (10, 1.0, 'b')),
    ([(10, 1.0, 'a'), (20, 2.0, 'b')], (20, 1.0, 'b')),
])
def test_contains_no_detection(existing, new):
    existing = create_exclusion_set(existing, hp_ranges)
    new = hp_ranges.tuple_to_config(new)
    assert not DuplicateDetectorNoDetection().contains(existing, new)
