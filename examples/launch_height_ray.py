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

from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.skopt import SkOptSearch
import numpy as np

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.schedulers.ray_scheduler import RayTuneScheduler
from syne_tune import Tuner
from syne_tune.config_space import randint
from syne_tune import StoppingCriterion

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    random_seed = 31415927
    max_steps = 100
    n_workers = 4

    config_space = {
        "steps": max_steps,
        "width": randint(0, 20),
        "height": randint(-100, 100)
    }
    entry_point = str(
        Path(__file__).parent / "training_scripts" / "height_example" /
        "train_height.py")
    mode = "min"
    metric = "mean_loss"

    # Local back-end
    trial_backend = LocalBackend(entry_point=entry_point)

    # Hyperband scheduler with SkOpt searcher
    np.random.seed(random_seed)
    ray_searcher = SkOptSearch()
    ray_searcher.set_search_properties(
        mode=mode, metric=metric,
        config=RayTuneScheduler.convert_config_space(config_space))

    ray_scheduler = AsyncHyperBandScheduler(
        max_t=max_steps,
        time_attr="step",
        mode=mode,
        metric=metric)

    scheduler = RayTuneScheduler(
        config_space=config_space,
        ray_scheduler=ray_scheduler,
        ray_searcher=ray_searcher)

    stop_criterion = StoppingCriterion(max_wallclock_time=30)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
    )

    tuner.run()
