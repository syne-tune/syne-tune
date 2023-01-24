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
from pathlib import Path
import pytest

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint
from syne_tune.optimizer.baselines import (
    RandomSearch,
    GridSearch,
    BayesianOptimization,
    ASHA,
    HyperTune,
    SyncHyperband,
    DEHB,
    BOHB,
    SyncBOHB,
    BORE,
    KDE,
    MOBSTER,
    PASHA,
    SyncMOBSTER,
    ASHABORE,
    BOTorch,
    REA,
    ConstrainedBayesianOptimization,
    KDE,
)

# List of schedulers to test, and whether they are multi-fidelity
# Does not test transfer learning schedulers
SCHEDULERS = [
    (GridSearch, False),
    (RandomSearch, False),
    (BayesianOptimization, False),
    (ASHA, True),
    (HyperTune, True),
    (SyncHyperband, True),
    (DEHB, True),
    (BOHB, True),
    (SyncBOHB, True),
    (BORE, False),
    (KDE, False),
    (MOBSTER, True),
    (PASHA, True),
    (SyncMOBSTER, True),
    (ASHABORE, True),
    (BOTorch, False),
    (REA, False),
    (ConstrainedBayesianOptimization, False),
    (KDE, False),
]


@pytest.mark.timeout(20)
@pytest.mark.parametrize("Scheduler, mul_fid", SCHEDULERS)
def test_points_to_evaluate(Scheduler, mul_fid):

    # Use train_height backend for our tests
    entry_point = str(
        Path(__file__).parent.parent
        / "examples"
        / "training_scripts"
        / "height_example"
        / "train_height.py"
    )
    max_steps = 5
    config_space = {
        "steps": max_steps,
        "width": randint(0, 20),
        "height": randint(-100, 100),
    }
    metric = "mean_loss"
    n_workers = 4

    # The points we require the schedulers to start by evaluating
    points_to_evaluate = [
        {"steps": max_steps, "width": 3, "height": 0},
        {"steps": max_steps, "width": 13, "height": -20},
        {"steps": max_steps, "width": 19, "height": 30},
    ]

    # The scheduler specification varies depending on the type of scheduler
    if Scheduler is ConstrainedBayesianOptimization:
        scheduler = Scheduler(
            config_space=config_space,
            metric=metric,
            points_to_evaluate=points_to_evaluate,
            constraint_attr="height",
        )
    elif mul_fid:
        scheduler = Scheduler(
            config_space=config_space,
            metric=metric,
            points_to_evaluate=points_to_evaluate,
            resource_attr="epoch",
            max_resource_attr="steps",
        )
    else:
        scheduler = Scheduler(
            config_space=config_space,
            metric=metric,
            points_to_evaluate=points_to_evaluate,
        )

    # Set up tuner and run for a few evaluations
    tuner = Tuner(
        trial_backend=LocalBackend(entry_point=entry_point),
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_num_evaluations=5),
        n_workers=n_workers,
        sleep_time=0.001,
    )
    tuner.run()

    # Extract the results
    df = tuner.tuning_status.get_dataframe()

    # Check that the first points match those defined in points_to_evaluate
    for ii in range(len(points_to_evaluate)):
        for key in points_to_evaluate[ii]:
            if not mul_fid or key != "steps":
                assert df[key][ii] == points_to_evaluate[ii][key], (
                    "Initial point %s does not match listed points_to_evaluate." % ii
                )
