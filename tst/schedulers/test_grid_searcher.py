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
from syne_tune.config_space import randint, uniform, choice
from syne_tune.optimizer.schedulers.searchers import GridSearcher

config_space = {
    "char_attr": choice(["a", "b"]),
    "int_attr": choice([1, 2]),
}
all_candidates_on_grid = [
    {"char_attr": "a", "int_attr": 1},
    {"char_attr": "b", "int_attr": 1},
    {"char_attr": "a", "int_attr": 2},
    {"char_attr": "b", "int_attr": 2},
]


def test_get_config():
    config_spaces = [
        {"single_attr": choice(["a", "b", "c", "d"])},
        {
            "attr_with_duplicates": choice([1, 1, 2, 2]),
            "other_attr": choice(["a", "b"]),
        },
    ]
    num_valid_config = 4

    for config_space in config_spaces:
        searcher = GridSearcher(config_space, metric="accuracy", points_to_evaluate=[])
        for trial_id in range(num_valid_config):
            # These should get new config
            config = searcher.get_config(trial_id=trial_id)
            assert config is not None

        config = searcher.get_config(trial_id=trial_id)
        assert config is None


def test_generate_all_candidates_on_grid():
    searcher = GridSearcher(config_space, metric="accuracy", points_to_evaluate=[])
    for i in range(len(all_candidates_on_grid)):
        assert searcher.get_config(trial_id=i) in all_candidates_on_grid


def test_non_shuffle():
    searcher = GridSearcher(
        config_space, metric="accuracy", shuffle_config=False, points_to_evaluate=[]
    )
    for i in range(len(all_candidates_on_grid)):
        config = searcher.get_config(trial_id=i)
        assert config == all_candidates_on_grid[i]


def test_store_and_restore_state_without_initial_config():
    searcher = GridSearcher(
        config_space, metric="accuracy", points_to_evaluate=[], shuffle_config=False
    )
    previous_config = searcher.get_config(trial_id=0)
    state = searcher.get_state()
    new_searcher = searcher.clone_from_state(state)
    assert previous_config == all_candidates_on_grid[0]
    for i in range(1, len(all_candidates_on_grid)):
        new_config = new_searcher.get_config(trail_id=i)
        assert new_config == all_candidates_on_grid[i]


def test_store_and_restore_state_with_initial_config():
    inital_config = [
        {"char_attr": "a", "int_attr": 1},
        {"char_attr": "b", "int_attr": 2},
    ]
    searcher = GridSearcher(
        config_space,
        metric="accuracy",
        points_to_evaluate=inital_config,
        shuffle_config=False,
    )
    previous_config = searcher.get_config(trial_id=0)
    state = searcher.get_state()
    new_searcher = searcher.clone_from_state(state)
    assert previous_config == all_candidates_on_grid[0]
    for idx in [3, 1, 2]:
        new_config = new_searcher.get_config(trail_id=idx)
        assert new_config in all_candidates_on_grid


def test_grid_config():
    config_space = {
        "char_attr": choice(["a", "b"]),
        "float_attr": uniform(1, 5),
        "int_attr": randint(10, 40),
    }
    num_samples = {"float_attr": 2, "int_attr": 2}

    all_candidates_on_grid = [
        {"char_attr": "a", "float_attr": 2.0, "int_attr": 17},
        {"char_attr": "a", "float_attr": 2.0, "int_attr": 33},
        {"char_attr": "a", "float_attr": 4.0, "int_attr": 17},
        {"char_attr": "a", "float_attr": 4.0, "int_attr": 33},
        {"char_attr": "b", "float_attr": 2.0, "int_attr": 17},
        {"char_attr": "b", "float_attr": 2.0, "int_attr": 33},
        {"char_attr": "b", "float_attr": 4.0, "int_attr": 17},
        {"char_attr": "b", "float_attr": 4.0, "int_attr": 33},
    ]

    num_valid_config = len(all_candidates_on_grid)
    searcher = GridSearcher(
        config_space, num_samples=num_samples, metric="accuracy", points_to_evaluate=[]
    )
    for trial_id in range(num_valid_config):
        # These should get new config
        config = searcher.get_config(trial_id=trial_id)
        assert config in all_candidates_on_grid
