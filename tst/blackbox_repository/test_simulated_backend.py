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
import numpy as np
import pandas as pd

from syne_tune.config_space import randint
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular
from syne_tune.blackbox_repository.simulated_tabular_backend import (
    UserBlackboxBackend,
)


n = 10
x1 = np.arange(n)
x2 = np.arange(n)[::-1]
hp_names = ["hp_x1", "hp_x2"]

cs = {name: randint(0, n - 1) for name in hp_names}

n_epochs = 5
resource_attr = "hp_epoch"
cs_fidelity = {
    resource_attr: randint(1, n_epochs),
}


def test_pause_and_resume():
    data = np.stack([x1, x2]).T
    hyperparameters = pd.DataFrame(data=data, columns=hp_names)
    num_seeds = 1
    metric = "error"
    elapsed_time_attr = "elapsed_time"
    objective_names = [metric, elapsed_time_attr]
    num_objectives = len(objective_names)
    objectives_evaluations = np.random.rand(
        len(hyperparameters), num_seeds, n_epochs, num_objectives
    )
    objectives_evaluations[:, :, :, 1] = np.cumsum(
        np.abs(objectives_evaluations[:, :, :, 1]), axis=2
    )
    blackbox = BlackboxTabular(
        hyperparameters=hyperparameters,
        configuration_space=cs,
        fidelity_space=cs_fidelity,
        objectives_evaluations=objectives_evaluations,
        objectives_names=objective_names,
    )
    backend = UserBlackboxBackend(
        blackbox=blackbox,
        elapsed_time_attr=elapsed_time_attr,
    )

    inds = [2, 4]
    configs = [dict(zip(hp_names, list(data[ind]))) for ind in inds]
    metrics = [objectives_evaluations[ind, 0, :, 0] for ind in inds]
    elapsed_times = [objectives_evaluations[ind, 0, :, 1] for ind in inds]
    # Start 2 trials
    backend.time_keeper.start_of_time()
    for config in configs:
        backend.start_trial(config)
    # Advance time and read out results
    pause_resources = [1, 2]
    step = max(elapsed_times[0][-1], elapsed_times[1][-1])
    print(f"elapsed_times = {elapsed_times}, step = {step}")
    backend.time_keeper.advance(step)
    _, results = backend.fetch_status_results(trial_ids=[0, 1])
    num_found = [0, 0]
    for trial_id, result in results:
        resource = result[resource_attr]
        fval = result[metric]
        elapsed_time = result[elapsed_time_attr]
        assert fval == metrics[trial_id][resource - 1]
        assert elapsed_time == elapsed_times[trial_id][resource - 1]
        if resource <= pause_resources[trial_id]:
            num_found[trial_id] += 1
    assert all(x == y for x, y in zip(pause_resources, num_found)), (
        pause_resources,
        num_found,
    )
    # Pause trials
    results = [
        {
            metric: fval,
            elapsed_time_attr: elapsed_time,
            resource_attr: resource,
        }
        for fval, elapsed_time, resource in zip(metrics, elapsed_times, pause_resources)
    ]
    for trial_id, result in enumerate(results):
        backend.pause_trial(trial_id=trial_id, result=result)
    # Resume paused trials and check that they do not start from scratch
    for trial_id in range(2):
        backend.resume_trial(trial_id)
    backend.time_keeper.advance(step)
    _, results = backend.fetch_status_results(trial_ids=[0, 1])
    got_it = [False, False]
    for trial_id, result in results:
        resource = result[resource_attr]
        fval = result[metric]
        assert fval == metrics[trial_id][resource - 1]
        pause_resource = pause_resources[trial_id]
        assert resource > pause_resource
        if resource == pause_resource + 1:
            got_it[trial_id] = True
    assert all(got_it)
