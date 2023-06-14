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
from functools import partial
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint, uniform
from syne_tune.optimizer.baselines import BayesianOptimization
from syne_tune.optimizer.schedulers import FIFOScheduler
from syne_tune.optimizer.schedulers.multiobjective.multi_surrogate_multi_objective_searcher import (
    MultiObjectiveMultiSurrogateSearcher,
)
from syne_tune.optimizer.schedulers.multiobjective.random_scalarization import (
    MultiObjectiveLCBRandomLinearScalarization,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.estimator import Estimator


def create_gaussian_process_estimator(
    config_space: Dict[str, Any],
    metric: str,
    mode: Optional[str] = None,
    random_seed: Optional[int] = None,
    search_options: Optional[Dict[str, Any]] = None,
) -> Estimator:
    scheduler = BayesianOptimization(
        config_space=config_space,
        metric=metric,
        mode=mode,
        random_seed=random_seed,
        search_options=search_options,
    )
    searcher = scheduler.searcher  # GPFIFOSearcher
    state_transformer = searcher.state_transformer  # ModelStateTransformer
    estimator = state_transformer.estimator  # GaussProcEmpiricalBayesEstimator

    # update the estimator properties
    estimator.active_metric = metric
    return estimator


def main():
    random_seed = 6287623
    # Hyperparameter configuration space
    config_space = {
        "steps": randint(0, 100),
        "theta": uniform(0, np.pi / 2),
        "sleep_time": 0.01,
    }
    metrics = ["y1", "y2"]
    modes = ["min", "min"]
    # Create Gaussian process estimators
    # In ``search_options``, the GP model can be configured, see comments
    # of ``GPFIFOSearcher``
    search_options = {"debug_log": False, "no_fantasizing": True}
    myestimators = {
        metric: create_gaussian_process_estimator(
            config_space=config_space,
            metric=metric,
            mode=mode,
            search_options=search_options,
        )
        for metric, mode in zip(metrics, modes)
    }
    searcher = MultiObjectiveMultiSurrogateSearcher(
        config_space=config_space,
        metric=metrics,
        estimators=myestimators,
        scoring_class=partial(
            MultiObjectiveLCBRandomLinearScalarization, random_seed=random_seed
        ),
        random_seed=random_seed,
    )

    scheduler = FIFOScheduler(
        config_space,
        metric=metrics,
        mode=modes,
        searcher=searcher,
        search_options=search_options,
    )

    entry_point = str(
        Path(__file__).parent
        / "training_scripts"
        / "mo_artificial"
        / "mo_artificial.py"
    )
    tuner = Tuner(
        trial_backend=LocalBackend(entry_point=entry_point),
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_wallclock_time=30),
        n_workers=1,  # how many trials are evaluated in parallel
    )
    tuner.run()


if __name__ == "__main__":
    main()
