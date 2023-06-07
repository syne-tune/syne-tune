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
from typing import Optional, Tuple
import tempfile
import time

import numpy as np

from syne_tune.backend import LocalBackend
from syne_tune.backend.trial_status import Status
from syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn.estimator import (
    SKLearnEstimator,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn.predictor import (
    SKLearnPredictor,
)
from syne_tune.util import script_height_example_path
from syne_tune.blackbox_repository.simulated_tabular_backend import (
    UserBlackboxBackend,
)
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune import Tuner
from syne_tune import StoppingCriterion
from examples.training_scripts.height_example.train_height import (
    height_config_space,
    RESOURCE_ATTR,
    METRIC_ATTR,
    METRIC_MODE,
    MAX_RESOURCE_ATTR,
)
from examples.training_scripts.height_example.blackbox_height import (
    HeightExampleBlackbox,
)


class TestPredictor(SKLearnPredictor):
    def predict(
        self, X: np.ndarray, return_std: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        nexamples = X.shape[0]
        return np.ones(nexamples), np.ones(nexamples)


class TestEstimator(SKLearnEstimator):
    def fit(self, X: np.ndarray, y: np.ndarray, update_params: bool) -> TestPredictor:
        return TestPredictor()


def temporary_local_backend(entry_point: str, **kwargs):
    """
    :param entry_point:
    :return: a backend whose files are deleted after finishing to avoid side-effects. This is used in unit-tests.
    """
    with tempfile.TemporaryDirectory() as local_path:
        backend = LocalBackend(entry_point=entry_point, **kwargs)
        backend.set_path(results_root=local_path)
        return backend


def wait_until_all_trials_completed(backend):
    def all_status(backend, trial_ids):
        return [trial.status for trial in backend._all_trial_results(trial_ids)]

    i = 0
    while not all(
        [
            status == Status.completed
            for status in all_status(backend, backend.trial_ids)
        ]
    ):
        time.sleep(0.1)
        i += 1
        assert i < 100, "backend trials did not finish after 10s"


def run_experiment_with_height(
    make_scheduler: callable,
    simulated: bool,
    mode: Optional[str] = None,
    config_space: Optional[dict] = None,
    **kwargs,
):
    random_seed = 382378624
    if mode is None:
        mode = METRIC_MODE

    if simulated:
        max_steps = 9
        num_workers = 4
        script_sleep_time = 0.1
        tuner_sleep_time = 0
        max_wallclock_time = 30
        callbacks = [SimulatorCallback()]
    else:
        max_steps = 5
        num_workers = 2
        script_sleep_time = 0.001
        tuner_sleep_time = 0.1
        max_wallclock_time = 0.2
        callbacks = None
    if "max_wallclock_time" in kwargs:
        max_wallclock_time = kwargs["max_wallclock_time"]
    if "num_workers" in kwargs:
        num_workers = kwargs["num_workers"]

    # It is possible to pass ``config_space`` other than the default one
    if config_space is None:
        config_space = height_config_space(
            max_steps, sleep_time=script_sleep_time if not simulated else None
        )
    entry_point = str(script_height_example_path())
    metric = METRIC_ATTR

    if simulated:
        elapsed_time_attr = "elapsed_time"
        blackbox = HeightExampleBlackbox(
            max_steps=max_steps,
            sleep_time=script_sleep_time,
            elapsed_time_attr=elapsed_time_attr,
        )
        trial_backend = UserBlackboxBackend(
            blackbox=blackbox,
            elapsed_time_attr=elapsed_time_attr,
            max_resource_attr=MAX_RESOURCE_ATTR,
        )
    else:
        trial_backend = temporary_local_backend(entry_point=entry_point)

    myscheduler = make_scheduler(
        config_space,
        metric=metric,
        mode=mode,
        random_seed=random_seed,
        resource_attr=RESOURCE_ATTR,
        max_resource_attr=MAX_RESOURCE_ATTR,
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=myscheduler,
        n_workers=num_workers,
        stop_criterion=stop_criterion,
        sleep_time=tuner_sleep_time,
        callbacks=callbacks,
        save_tuner=False,
    )
    tuner.run()
