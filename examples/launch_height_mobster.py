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
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.tuner import Tuner
from syne_tune.search_space import randint
from syne_tune.stopping_criterion import StoppingCriterion


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
    backend = LocalBackend(entry_point=entry_point)

    # MOBSTER: Hyperband scheduling with GP-BO search
    # Note: HyperbandScheduler requires resource_attr to have
    # positive int values, so we use 'epoch'
    searcher = 'bayesopt'
    search_options = {'num_init_random': n_workers + 2}
    scheduler = HyperbandScheduler(
        config_space,
        searcher=searcher,
        search_options=search_options,
        max_t=max_steps,
        resource_attr='epoch',
        mode=mode,
        metric=metric,
        random_seed=random_seed)

    stop_criterion = StoppingCriterion(max_wallclock_time=30)
    tuner = Tuner(
        backend=backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
    )

    tuner.run()
