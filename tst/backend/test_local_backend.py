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
import time
import logging
from pathlib import Path

from syne_tune.backend.trial_status import Status
from syne_tune.util import script_checkpoint_example_path
from tst.util_test import temporary_local_backend


def check_metrics(metrics_observed, metrics_expected):
    assert len(metrics_observed) == len(metrics_expected)
    for (trial_id1, result1), (trial_id2, result2) in zip(metrics_observed, metrics_expected):
        assert trial_id1 == trial_id2
        for key in ['step', 'train_acc']:
            assert result1[key] == result2[key]


def status(backend, trial_ids):
    return [trial.status for trial in backend._all_trial_results(trial_ids)]


def wait_until_all_trials_completed(backend):
    i = 0
    while not all([status == Status.completed for status in status(backend, backend.trial_ids)]):
        time.sleep(0.1)
        assert i < 100, "backend trials did not finish after 10s"


def get_status_metrics(backend, trial_id):
    trial_status_dict, new_metrics = backend.fetch_status_results([trial_id])
    trial_statuses = {trial_id: status for (trial_id, (_, status)) in trial_status_dict.items()}
    return trial_statuses, new_metrics


def test_local_backend_checkpoint(caplog):
    caplog.set_level(logging.INFO)
    path_script = script_checkpoint_example_path()
    backend = temporary_local_backend(entry_point=path_script)
    trial_id = backend.start_trial(config={'num-epochs': 2}).trial_id
    wait_until_all_trials_completed(backend)

    trial_statuses, new_metrics = get_status_metrics(backend, trial_id)
    assert trial_statuses == {trial_id: Status.completed}
    check_metrics(new_metrics, [(trial_id, {'step': 0, 'train_acc': 1}), (trial_id, {'step': 1, 'train_acc': 2})])

    trial_statuses, new_metrics = get_status_metrics(backend, trial_id)
    check_metrics(new_metrics, [])

    trial_statuses, new_metrics = get_status_metrics(backend, trial_id)
    assert new_metrics == []

    backend.pause_trial(trial_id=trial_id)
    backend.resume_trial(trial_id=trial_id)

    wait_until_all_trials_completed(backend)
    trial_statuses, new_metrics = get_status_metrics(backend, trial_id)
    assert trial_statuses == {trial_id: Status.completed}
    check_metrics(new_metrics, [])

    trial_statuses, new_metrics = get_status_metrics(backend, trial_id)
    assert new_metrics == []

    trial_id = backend.start_trial(config={'num-epochs': 200}).trial_id
    backend.stop_trial(trial_id=trial_id)

    trial_statuses, new_metrics = get_status_metrics(backend, trial_id)
    assert trial_statuses == {trial_id: Status.stopped}

    trial_id = backend.start_trial(config={'num-epochs': 200}).trial_id
    backend.pause_trial(trial_id=trial_id)
    trial_statuses, new_metrics = get_status_metrics(backend, trial_id)
    assert trial_statuses == {trial_id: Status.paused}


def test_resume_config_local_backend(caplog):
    caplog.set_level(logging.INFO)
    path_script = script_checkpoint_example_path()
    backend = temporary_local_backend(entry_point=path_script)
    trial_id = backend.start_trial(config={'num-epochs': 2}).trial_id

    wait_until_all_trials_completed(backend)

    trial_statuses, new_metrics = get_status_metrics(backend, trial_id)
    assert trial_statuses == {trial_id: Status.completed}
    check_metrics(new_metrics, [(trial_id, {'step': 0, 'train_acc': 1}), (trial_id, {'step': 1, 'train_acc': 2})])

    backend.pause_trial(trial_id=trial_id)
    backend.resume_trial(trial_id, new_config={'num-epochs': 4})

    wait_until_all_trials_completed(backend)

    trial_statuses, new_metrics = get_status_metrics(backend, trial_id)
    assert trial_statuses == {trial_id: Status.completed}
    check_metrics(new_metrics, [
        (trial_id, {'step': 2, 'train_acc': 3}),
        (trial_id, {'step': 3, 'train_acc': 4}),
    ])


def test_start_config_previous_checkpoint(caplog):
    caplog.set_level(logging.INFO)
    path_script = Path(__file__).parent / "main_checkpoint.py"
    backend = temporary_local_backend(entry_point=path_script)

    # we start two trials, the checkpoint content should be resp. state-0, state-1 and then reported.
    backend.start_trial(config={"name": "state-0"})
    backend.start_trial(config={"name": "state-1"})

    wait_until_all_trials_completed(backend)
    # we check whether a trial can be started from a previous checkpoint
    backend.start_trial(checkpoint_trial_id=0, config={"name": "state-2"})
    backend.start_trial(checkpoint_trial_id=1, config={"name": "state-3"})

    wait_until_all_trials_completed(backend)

    # the two trials that were started should have reported the content of their checkpoint, e.g. state-0 and state-1.
    _, new_metrics = backend.fetch_status_results([0, 1, 2, 3])
    results = list(sorted(new_metrics, key=lambda x: x[0]))
    results = [res['checkpoint_content'] for _, res in results]
    assert results == ['nothing', 'nothing', 'state-0', 'state-1']


def test_gpu_allocation(caplog):
    caplog.set_level(logging.INFO)
    path_script = Path(__file__).parent / "main_checkpoint.py"
    backend = temporary_local_backend(entry_point=path_script)

    backend._prepare_for_schedule(num_gpus=4)
    env = dict()
    for trial_id in range(4):
        backend._allocate_gpu(trial_id=trial_id, env=env)
    for trial_id in range(4):
        gpu = trial_id
        assert backend.trial_gpu[trial_id] == gpu
        assert backend.gpu_times_assigned[gpu] == 1
    backend._deallocate_gpu(trial_id=2)
    backend._allocate_gpu(trial_id=4, env=env)
    # GPU 2 is the only free one, so must be used:
    assert backend.trial_gpu[4] == 2
    assert backend.gpu_times_assigned[2] == 2
    backend._deallocate_gpu(trial_id=0)
    backend._deallocate_gpu(trial_id=2)
    backend._allocate_gpu(trial_id=5, env=env)
    # Both GPUs 0, 2 are free, but 0 has less prior assignments:
    assert backend.trial_gpu[5] == 0
    assert backend.gpu_times_assigned[0] == 2
    backend._allocate_gpu(trial_id=6, env=env)
    # All GPUs are allocated. 1, 3 have less prior assignments:
    gpu = backend.trial_gpu[6]
    assert gpu in {1, 3}
    assert backend.gpu_times_assigned[gpu] == 2
