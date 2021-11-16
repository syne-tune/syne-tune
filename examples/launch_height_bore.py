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
import logging
from pathlib import Path

from syne_tune.backend.local_backend import LocalBackend
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.searchers.bore import Bore
from syne_tune.tuner import Tuner
from syne_tune.search_space import randint
from syne_tune.stopping_criterion import StoppingCriterion


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    max_trials = 100
    n_workers = 1  # BORE only works for the non-parallel setting so far

    config_space = {
        'steps': 100,
        "width": randint(0, 20),
        "height": randint(-100, 100)
    }
    entry_point = Path(__file__).parent / "training_scripts" / "height_example" / "train_height.py"
    mode = "min"
    metric = "mean_loss"

    backend = LocalBackend(entry_point=str(entry_point))

    bore = Bore(config_space=config_space, metric=metric, mode=mode)
    scheduler = FIFOScheduler(
        config_space,
        searcher=bore,
        mode=mode,
        metric=metric)

    tuner = Tuner(
        backend=backend,
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_wallclock_time=120),
        n_workers=n_workers,
    )

    tuner.run()
