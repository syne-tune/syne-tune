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

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.baselines import (
    RandomSearch,
    ASHA,
)

# from syne_tune.optimizer.baselines import PASHA, BORE  # noqa: F401
# from syne_tune.optimizer.schedulers.synchronous import \
#    SynchronousGeometricHyperbandScheduler  # noqa: F401
# from syne_tune.optimizer.schedulers import FIFOScheduler  # noqa: F401
# from syne_tune.optimizer.schedulers.botorch.botorch_searcher import BotorchSearcher  # noqa: F401
from syne_tune import Tuner, StoppingCriterion
from syne_tune.config_space import randint
from syne_tune.try_import import try_import_gpsearchers_message


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    random_seed = 31415927
    max_steps = 100
    n_workers = 4

    config_space = {
        "steps": max_steps,
        "width": randint(0, 20),
        "height": randint(-100, 100),
    }
    entry_point = (
        Path(__file__).parent
        / "training_scripts"
        / "height_example"
        / "train_height.py"
    )
    mode = "min"
    metric = "mean_loss"

    schedulers = [
        RandomSearch(config_space, metric=metric, mode=mode),
        ASHA(
            config_space,
            metric=metric,
            resource_attr="epoch",
            max_t=max_steps,
            mode=mode,
        ),
        # Commented as needs extra libraries or to save CI testing time. Since we are testing those baselines
        # in our baseline, we keep the uncommented list of schedulers to a small number.
        # PASHA(config_space, metric=metric, resource_attr='epoch', max_t=max_steps, mode=mode),
        # BORE(config_space, metric=metric, mode=mode),
        # SynchronousGeometricHyperbandScheduler(
        #     config_space,
        #     max_resource_level=max_steps,
        #     brackets=3,
        #     max_resource_attr='steps',
        #     resource_attr='epoch',
        #     batch_size=n_workers,
        #     mode=mode,
        #     metric=metric,
        # ),
        # FIFOScheduler(
        #     config_space,
        #     searcher=BotorchSearcher(config_space=config_space, metric=metric, mode='min'),
        #     metric=metric
        # ),
    ]
    try:
        from syne_tune.optimizer.baselines import BayesianOptimization

        # example of setting additional kwargs arguments
        schedulers.append(
            BayesianOptimization(
                config_space,
                metric=metric,
                mode=mode,
                search_options={"num_init_random": n_workers + 2},
            )
        )
        from syne_tune.optimizer.baselines import MOBSTER

        schedulers.append(
            MOBSTER(
                config_space,
                metric=metric,
                resource_attr="epoch",
                max_t=max_steps,
                mode=mode,
            )
        )
    except Exception:
        logging.info(try_import_gpsearchers_message())

    for scheduler in schedulers:
        logging.info(f"\n*** running scheduler {scheduler} ***\n")

        trial_backend = LocalBackend(entry_point=str(entry_point))

        stop_criterion = StoppingCriterion(
            max_wallclock_time=5, min_metric_value={"mean_loss": -6.0}
        )
        tuner = Tuner(
            trial_backend=trial_backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=n_workers,
        )

        tuner.run()
