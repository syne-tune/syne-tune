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
from syne_tune.optimizer.schedulers.searchers.searcher import \
    RandomSearcher
from syne_tune.config_space import choice, randint


def test_no_duplicates():
    config_spaces = [
        {'cat_attr': choice(['a', 'b'])},
        {'int_attr': randint(lower=0, upper=1)},
    ]
    num_suggest_to_fail = 3

    for config_space in config_spaces:
        searcher = RandomSearcher(config_space, metric='accuracy')
        for trial_id in range(num_suggest_to_fail):
            # These should not fail
            config = searcher.get_config(trial_id=trial_id)
            if trial_id < num_suggest_to_fail - 1:
                assert config is not None
            else:
                assert config is None
