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

# If you like to run the code linked in this tutorial, please make sure to use
# the current `PyPI` release. If you cloned the source repository, this is
# obtained as follows:
#
# ```bash
# git checkout -b basic_tutorial v0.11
# ```
#
# This gives you a local branch `basic_tutorial`, in which you can play around
# with the code.
import logging
from pathlib import Path

from syne_tune.config_space import randint, uniform, loguniform


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # [1]
    random_seed = 31415927
    n_workers = 4
    max_wallclock_time = 3 * 3600  # Run for 3 hours
    max_resource_level = 81  # Maximum number of training epochs

    # Here, we specify the training script we want to tune
    # - `mode` and `metric` must match what is reported in the training script
    # [2]
    entry_point = str(Path(__file__).parent / "traincode_report_end.py")
    mode = 'max'
    metric = 'accuracy'
    max_resource_attr = 'epochs'

    # Search space (or configuration space)
    # For each tunable parameter, need to define type, range, and encoding
    # (linear, logarithmic)
    # [3]
    config_space = {
        'n_units_1': randint(4, 1024),
        'n_units_2': randint(4, 1024),
        'batch_size': randint(8, 128),
        'dropout_1': uniform(0, 0.99),
        'dropout_2': uniform(0, 0.99),
        'learning_rate': loguniform(1e-6, 1),
        'weight_decay': loguniform(1e-8, 1),
    }
