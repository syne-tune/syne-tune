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
An example showing to launch a tuning of a python function `train_height`.
"""

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend.python_backend import PythonBackend
from syne_tune.optimizer.baselines import ASHA
from examples.training_scripts.height_example.train_height import (
    height_config_space,
    train_height as train_height_inner,
    RESOURCE_ATTR,
    METRIC_ATTR,
    METRIC_MODE,
)


def train_height(steps: int, width: float, height: float):
    """
    The function to be tuned, note that import must be in PythonBackend and no global variable are allowed,
    more details on requirements of tuned functions can be found in `PythonBackend`.
    """
    import logging
    from syne_tune import Reporter
    import time

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    reporter = Reporter()
    for step in range(steps):
        dummy_score = train_height_inner(step, width, height)
        # Feed the score back to Syne Tune.
        reporter(
            **{
                "step": step,
                METRIC_ATTR: dummy_score,
                RESOURCE_ATTR: step + 1,
            }
        )
        time.sleep(0.1)


if __name__ == "__main__":
    import logging

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    max_steps = 100
    n_workers = 4

    config_space = height_config_space(max_steps)
    mode = METRIC_MODE
    metric = METRIC_ATTR

    scheduler = ASHA(
        config_space,
        metric=METRIC_ATTR,
        resource_attr=RESOURCE_ATTR,
        max_t=max_steps,
        mode=METRIC_MODE,
    )

    trial_backend = PythonBackend(tune_function=train_height, config_space=config_space)

    stop_criterion = StoppingCriterion(
        max_wallclock_time=10, min_metric_value={METRIC_ATTR: -6.0}
    )
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
    )
    tuner.run()
