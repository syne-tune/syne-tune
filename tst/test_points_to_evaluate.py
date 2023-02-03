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
import sys

from examples.training_scripts.height_example.train_height import (
    height_config_space,
    METRIC_ATTR,
    RESOURCE_ATTR,
    MAX_RESOURCE_ATTR,
)
from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.optimizer.baselines import (
    GridSearch,
    RandomSearch,
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
    REA,
    ConstrainedBayesianOptimization,
    KDE,
)

# List of schedulers to test, and whether they are multi-fidelity or constrained
# Does not test transfer learning schedulers
SCHEDULERS = [
    (GridSearch, False, False),
    (RandomSearch, False, False),
    (BayesianOptimization, False, False),
    (ASHA, True, False),
    (HyperTune, True, False),
    (SyncHyperband, True, False),
    (DEHB, True, False),
    (BOHB, True, False),
    (SyncBOHB, True, False),
    (BORE, False, False),
    (KDE, False, False),
    (MOBSTER, True, False),
    (PASHA, True, False),
    (SyncMOBSTER, True, False),
    (ASHABORE, True, False),
    (REA, False, False),
    (ConstrainedBayesianOptimization, False, True),
    (KDE, False, False),
]
if sys.version_info >= (3, 8):
    # BoTorch scheduler requires Python 3.8 or later
    from syne_tune.optimizer.baselines import BoTorch

    SCHEDULERS.append((BoTorch, False, False))


@pytest.mark.timeout(20)
@pytest.mark.parametrize("scheduler_cls, mul_fid, constr", SCHEDULERS)
def test_points_to_evaluate(scheduler_cls, mul_fid, constr):

    # Use train_height backend for our tests
    entry_point = str(
        Path(__file__).parent.parent
        / "examples"
        / "training_scripts"
        / "height_example"
        / "train_height.py"
    )

    max_steps = 5
    # The points we require the schedulers to start by evaluating
    points_to_evaluate = [
        {"steps": max_steps, "width": 3, "height": 0},
        {"steps": max_steps, "width": 13, "height": -20},
        {"steps": max_steps, "width": 19, "height": 30},
    ]

    # The scheduler specification varies depending on the type of scheduler
    kwargs = {
        "config_space": height_config_space(max_steps=max_steps),
        "metric": METRIC_ATTR,
        "points_to_evaluate": points_to_evaluate,
    }

    if constr:
        kwargs["constraint_attr"] = "height"
    if mul_fid:
        kwargs["resource_attr"] = RESOURCE_ATTR
        kwargs["max_resource_attr"] = MAX_RESOURCE_ATTR

    # Set up tuner and run for a few evaluations
    tuner = Tuner(
        trial_backend=LocalBackend(entry_point=entry_point),
        scheduler=scheduler_cls(**kwargs),
        stop_criterion=StoppingCriterion(max_num_evaluations=5),
        n_workers=4,
        sleep_time=0.001,
    )
    tuner.run()

    # Extract the results
    df = tuner.tuning_status.get_dataframe()

    # Check that the first points match those defined in points_to_evaluate
    for ii in range(len(points_to_evaluate)):
        for key in points_to_evaluate[ii]:
            if not mul_fid or key != MAX_RESOURCE_ATTR:
                # Multi-fidelity schedulers might not do as many steps of the point
                assert df[key][ii] == points_to_evaluate[ii][key], (
                    "Initial point %s does not match listed points_to_evaluate." % ii
                )
