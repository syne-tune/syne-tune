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
from numpy.random import RandomState


RANDOM_SEED_UPPER_BOUND = 2**31 - 1


def generate_random_seed(random_state: RandomState = np.random) -> int:
    return random_state.randint(0, RANDOM_SEED_UPPER_BOUND)


class RandomSeedGenerator:
    def __init__(self, master_seed: int):
        self._random_state = np.random.RandomState(master_seed)

    def __call__(self) -> int:
        return generate_random_seed(self._random_state)
