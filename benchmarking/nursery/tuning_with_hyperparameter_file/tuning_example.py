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
"""
Example showing how to tune given a script ("training_script.py") that takes input hyperparameters
as a file rather than command line arguments.
Note that this approach only works with `LocalBackend` at the moment.
"""
from pathlib import Path

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint
from syne_tune.optimizer.baselines import RandomSearch

if __name__ == "__main__":
    config_space = {"x": randint(0, 10)}
    tuner = Tuner(
        scheduler=RandomSearch(config_space=config_space, metric="error"),
        trial_backend=LocalBackend(
            entry_point=str(Path(__file__).parent / "training_script.py")
        ),
        stop_criterion=StoppingCriterion(max_wallclock_time=20),
        n_workers=2,
    )
    tuner.run()
