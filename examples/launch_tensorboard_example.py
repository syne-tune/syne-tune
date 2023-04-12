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
Example showing how to visualize the HPO process of Syne Tune with Tensorboard.
Results will be stored in ~/syne-tune/{tuner_name}/tensoboard_output. To start
tensorboard, execute in a separate shell:

.. code:: bash

   tensorboard --logdir  /~/syne-tune/{tuner_name}/tensorboard_output

Open the displayed URL in the browser.

To use this functionality you need to install tensorboardX:

.. code:: bash

   pip install tensorboardX

"""

import logging
from pathlib import Path

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune import Tuner, StoppingCriterion
from syne_tune.config_space import randint
from syne_tune.callbacks.tensorboard_callback import TensorboardCallback
from syne_tune.results_callback import StoreResultsCallback
from examples.training_scripts.height_example.train_height import (
    METRIC_ATTR,
    METRIC_MODE,
    MAX_RESOURCE_ATTR,
)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    random_seed = 31415927
    max_steps = 100
    n_workers = 4

    config_space = {
        MAX_RESOURCE_ATTR: max_steps,
        "width": randint(0, 20),
        "height": randint(-100, 100),
    }
    entry_point = str(
        Path(__file__).parent
        / "training_scripts"
        / "height_example"
        / "train_height.py"
    )

    trial_backend = LocalBackend(entry_point=entry_point)

    # Random search without stopping
    scheduler = RandomSearch(
        config_space, mode=METRIC_MODE, metric=METRIC_ATTR, random_seed=random_seed
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=20)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        n_workers=n_workers,
        stop_criterion=stop_criterion,
        results_update_interval=5,
        # Adding the TensorboardCallback overwrites the default callback which consists of the StoreResultsCallback.
        # To write results on this disk as well, we put this in here as well.
        callbacks=[
            TensorboardCallback(target_metric=METRIC_ATTR, mode=METRIC_MODE),
            StoreResultsCallback(),
        ],
        tuner_name="tensorboardx-demo",
        metadata={"description": "just an example"},
    )

    tuner.run()
