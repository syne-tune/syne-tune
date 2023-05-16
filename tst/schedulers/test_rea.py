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

from syne_tune.config_space import randint, choice
from syne_tune.optimizer.schedulers.searchers.regularized_evolution import (
    RegularizedEvolution,
)


def test_rea_config_space_size_one():
    config_space = {
        "a": randint(lower=1, upper=1),
        "b": choice(["a"]),
        "c": 25,
        "d": "dummy",
    }
    with pytest.raises(AssertionError):
        searcher = RegularizedEvolution(
            config_space=config_space,
            metric="error",
            mode="min",
            random_seed=314159,
        )
