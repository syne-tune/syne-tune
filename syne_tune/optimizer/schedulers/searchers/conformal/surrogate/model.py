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
from typing import Dict, Optional

import numpy as np
import pandas as pd


class Model:
    def __init__(
        self,
        config_space: Dict,
        mode: str,
        random_state: Optional[np.random.RandomState] = None,
    ):
        self.random_state = random_state if random_state else np.random
        self.config_space = config_space
        self.mode = mode
        self.config_candidates = []
        self.config_seen = set()

    def suggest(self) -> dict:
        raise NotImplementedError()

    def fit(self, df_features: pd.DataFrame, y: np.array):
        raise NotImplementedError()
