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
from syne_tune.config_space import *
from syne_tune.optimizer.schedulers.searchers.utils.common import Hyperparameter


def test_isinstance():
    config_space = {
        "char_attr": choice(["a", "b"]),
        "int_attr": choice([1, 2]),
        "float_attr": uniform(1, 5),
        "single_attr": 40,
    }
    assert isinstance(config_space["single_attr"], Hyperparameter)
    assert isinstance(config_space["char_attr"], Categorical)
