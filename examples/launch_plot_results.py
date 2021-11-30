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
import matplotlib.pyplot as plt

from syne_tune.backend.local_backend import LocalBackend
from syne_tune.constants import ST_TUNER_TIME
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
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

    backend = LocalBackend(entry_point=entry_point)

    # Random search without stopping
    scheduler = FIFOScheduler(
        config_space,
        searcher='random',
        mode=mode,
        metric=metric,
        random_seed=random_seed)

    stop_criterion = StoppingCriterion(max_wallclock_time=60)
    tuner = Tuner(
        backend=backend,
        scheduler=scheduler,
        n_workers=n_workers,
        stop_criterion=stop_criterion,
        results_update_interval=5,
        tuner_name="plot-results-demo",
        metadata={'description': 'just an example'},
    )

    tuner.run()

    tuning_experiment = load_experiment(tuner.name)
    print(tuning_experiment)
    df = tuning_experiment.results.sort_values(ST_TUNER_TIME)
    df.loc[:, 'best'] = df.loc[:, metric].cummin()
    df.plot(x=ST_TUNER_TIME, y="best")
    plt.xlabel("wallclock time")
    plt.ylabel(metric)
    plt.show()