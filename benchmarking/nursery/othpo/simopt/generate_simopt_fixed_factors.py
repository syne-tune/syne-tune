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
import pickle
from pathlib import Path
import os

cur_folder = str(Path(__file__).parent)
output_folder = cur_folder + "/generated_files"
os.makedirs(output_folder, exist_ok=True)

# Store default fixed factors
pickle.dump(
    {
        "mu": 1,
        "num_customer": 30,
        "num_prod": 3,
        "c_utility": [6, 8, 10],
        "init_level": [8.0, 6.0, 20.0],
        "price": [9.0, 9.0, 9.0],
        "cost": [5.0, 5.0, 5.0],
    },
    open(output_folder + "/default_fixed_factors.p", "wb"),
)
