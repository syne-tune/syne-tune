import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import pytest

from syne_tune.backend.trial_status import Status
from syne_tune.util import script_checkpoint_example_path
from tst.util_test import temporary_local_backend, wait_until_all_trials_completed


def check_metrics(metrics_observed, metrics_expected):
    assert len(metrics_observed) == len(metrics_expected)
    for (trial_id1, result1), (trial_id2, result2) in zip(
        metrics_observed, metrics_expected
    ):
        assert trial_id1 == trial_id2
        for key in ["epoch", "train_acc"]:
            assert result1[key] == result2[key]


def status(backend, trial_ids):
    return [trial.status for trial in backend._all_trial_results(trial_ids)]


def get_status_metrics(backend, trial_id):
    trial_status_dict, new_metrics = backend.fetch_status_results([trial_id])
    trial_statuses = {
        trial_id: status for (trial_id, (_, status)) in trial_status_dict.items()
    }
    return trial_statuses, new_metrics


@pytest.mark.timeout(7)
def test_local_backend_checkpoint(caplog):
    caplog.set_level(logging.INFO)
    path_script = script_checkpoint_example_path()
    backend = temporary_local_backend(entry_point=str(path_script))
    trial_id = backend.start_trial(config={"num-epochs": 2}).trial_id
    wait_until_all_trials_completed(backend)

    trial_statuses, new_metrics = get_status_metrics(backend, trial_id)
    assert trial_statuses == {trial_id: Status.completed}
    check_metrics(
        new_metrics,
        [
            (trial_id, {"epoch": 1, "train_acc": 1}),
            (trial_id, {"epoch": 2, "train_acc": 2}),
        ],
    )
    busy_trial_ids = backend.busy_trial_ids()
    assert len(busy_trial_ids) == 0, busy_trial_ids

    trial_statuses, new_metrics = get_status_metrics(backend, trial_id)
    check_metrics(new_metrics, [])

    trial_statuses, new_metrics = get_status_metrics(backend, trial_id)
    assert new_metrics == []

    backend.pause_trial(trial_id=trial_id)
    backend.resume_trial(trial_id=trial_id)

    busy_trial_ids = backend.busy_trial_ids()
    assert len(busy_trial_ids) == 1 and busy_trial_ids[0] == (
        trial_id,
        Status.in_progress,
    ), busy_trial_ids

    wait_until_all_trials_completed(backend)
    trial_statuses, new_metrics = get_status_metrics(backend, trial_id)
    assert trial_statuses == {trial_id: Status.completed}
    check_metrics(new_metrics, [])
    busy_trial_ids = backend.busy_trial_ids()
    assert len(busy_trial_ids) == 0, busy_trial_ids

    trial_statuses, new_metrics = get_status_metrics(backend, trial_id)
    assert new_metrics == []

    trial_id = backend.start_trial(config={"num-epochs": 200}).trial_id
    backend.stop_trial(trial_id=trial_id)

    trial_statuses, new_metrics = get_status_metrics(backend, trial_id)
    assert trial_statuses == {trial_id: Status.stopped}
    busy_trial_ids = backend.busy_trial_ids()
    assert len(busy_trial_ids) == 0, busy_trial_ids

    trial_id = backend.start_trial(config={"num-epochs": 200}).trial_id
    backend.pause_trial(trial_id=trial_id)
    trial_statuses, new_metrics = get_status_metrics(backend, trial_id)
    assert trial_statuses == {trial_id: Status.paused}
    busy_trial_ids = backend.busy_trial_ids()
    assert len(busy_trial_ids) == 0, busy_trial_ids


@pytest.mark.skip("Speed up as currently takes >7s")
def test_resume_config_local_backend(caplog):
    caplog.set_level(logging.INFO)
    path_script = script_checkpoint_example_path()
    backend = temporary_local_backend(entry_point=str(path_script))
    trial_id = backend.start_trial(config={"num-epochs": 2}).trial_id

    wait_until_all_trials_completed(backend)

    trial_statuses, new_metrics = get_status_metrics(backend, trial_id)
    assert trial_statuses == {trial_id: Status.completed}
    check_metrics(
        new_metrics,
        [
            (trial_id, {"epoch": 1, "train_acc": 1}),
            (trial_id, {"epoch": 2, "train_acc": 2}),
        ],
    )

    backend.pause_trial(trial_id=trial_id)
    backend.resume_trial(trial_id, new_config={"num-epochs": 4})

    wait_until_all_trials_completed(backend)

    trial_statuses, new_metrics = get_status_metrics(backend, trial_id)
    assert trial_statuses == {trial_id: Status.completed}
    check_metrics(
        new_metrics,
        [
            (trial_id, {"epoch": 3, "train_acc": 3}),
            (trial_id, {"epoch": 4, "train_acc": 4}),
        ],
    )


@pytest.mark.skip("Speed up as currently takes >7s")
def test_start_config_previous_checkpoint(caplog):
    caplog.set_level(logging.INFO)
    path_script = Path(__file__).parent / "main_checkpoint.py"
    backend = temporary_local_backend(entry_point=str(path_script))

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
    results = [res["checkpoint_content"] for _, res in results]
    assert results == ["nothing", "nothing", "state-0", "state-1"]


def _map_gpu(gpu: int, gpus_to_use: Optional[List[int]]) -> int:
    return gpu if gpus_to_use is None else gpus_to_use[gpu]


def _assert_cuda_visible_devices(
    env: Dict[str, Any],
    gpus_to_use: Optional[List[int]],
    visible_devices: List[int],
):
    cuda_visible_devices = ",".join(
        str(_map_gpu(gpu, gpus_to_use)) for gpu in visible_devices
    )
    assert "CUDA_VISIBLE_DEVICES" in env
    assert cuda_visible_devices == env["CUDA_VISIBLE_DEVICES"]


def test_gpu_allocation(caplog):
    caplog.set_level(logging.INFO)
    path_script = Path(__file__).parent / "main_checkpoint.py"

    for gpus_to_use, num_gpus in [
        (None, 4),
        ([5, 0, 7, 2], 8),
    ]:
        backend = temporary_local_backend(
            entry_point=str(path_script),
            gpus_to_use=gpus_to_use,
        )
        backend._prepare_for_schedule(num_gpus=num_gpus)
        env = dict()
        for trial_id in range(4):
            backend._allocate_gpu(trial_id=trial_id, env=env)
            _assert_cuda_visible_devices(env, gpus_to_use, [trial_id])
        for trial_id in range(4):
            gpu = trial_id
            assert backend.trial_gpu[trial_id] == [gpu]
            assert backend.gpu_times_assigned[gpu] == 1
        backend._deallocate_gpu(trial_id=2)
        backend._allocate_gpu(trial_id=4, env=env)
        # GPU 2 is the only free one, so must be used:
        assert backend.trial_gpu[4] == [2]
        _assert_cuda_visible_devices(env, gpus_to_use, backend.trial_gpu[4])
        assert backend.gpu_times_assigned[2] == 2
        backend._deallocate_gpu(trial_id=0)
        backend._deallocate_gpu(trial_id=2)
        backend._allocate_gpu(trial_id=5, env=env)
        # Both GPUs 0, 2 are free, but 0 has less prior assignments:
        assert backend.trial_gpu[5] == [0]
        _assert_cuda_visible_devices(env, gpus_to_use, backend.trial_gpu[5])
        assert backend.gpu_times_assigned[0] == 2
        backend._allocate_gpu(trial_id=6, env=env)
        # All GPUs are allocated. 1, 3 have less prior assignments:
        gpu = backend.trial_gpu[6]
        assert len(gpu) == 1
        gpu = gpu[0]
        assert gpu in {1, 3}
        _assert_cuda_visible_devices(env, gpus_to_use, [gpu])
        assert backend.gpu_times_assigned[gpu] == 2


def test_multi_gpu_allocation(caplog):
    caplog.set_level(logging.INFO)
    path_script = Path(__file__).parent / "main_checkpoint.py"

    for gpus_to_use, num_gpus in [
        (None, 8),
        ([10, 0, 15, 12, 1, 2, 13, 9], 16),
    ]:
        backend = temporary_local_backend(
            entry_point=str(path_script),
            num_gpus_per_trial=2,
            gpus_to_use=gpus_to_use,
        )
        backend._prepare_for_schedule(num_gpus=num_gpus)
        env = dict()
        for trial_id in range(4):
            backend._allocate_gpu(trial_id=trial_id, env=env)
            _assert_cuda_visible_devices(
                env, gpus_to_use, [2 * trial_id, 2 * trial_id + 1]
            )
        for trial_id in range(4):
            gpus = [2 * trial_id, 2 * trial_id + 1]
            assert backend.trial_gpu[trial_id] == gpus
            assert all(backend.gpu_times_assigned[gpu] == 1 for gpu in gpus)
        backend._deallocate_gpu(trial_id=2)
        backend._allocate_gpu(trial_id=4, env=env)
        # GPU 2 only free one, must have been used
        assert backend.trial_gpu[4] == [4, 5]
        _assert_cuda_visible_devices(env, gpus_to_use, backend.trial_gpu[4])
        assert backend.gpu_times_assigned[4] == 2
        assert backend.gpu_times_assigned[5] == 2
        backend._deallocate_gpu(trial_id=0)
        backend._deallocate_gpu(trial_id=2)
        backend._allocate_gpu(trial_id=5, env=env)
        # Both GPUs 0, 2 are free, but 0 has less prior assignments:
        assert backend.trial_gpu[5] == [0, 1]
        _assert_cuda_visible_devices(env, gpus_to_use, backend.trial_gpu[5])
        assert backend.gpu_times_assigned[0] == 2
        assert backend.gpu_times_assigned[1] == 2
        backend._allocate_gpu(trial_id=6, env=env)
        # All GPUs are allocated. 1, 3 have less prior assignments:
        gpus = backend.trial_gpu[6]
        assert gpus == [2, 3] or gpus == [6, 7]
        _assert_cuda_visible_devices(env, gpus_to_use, gpus)
        assert all(backend.gpu_times_assigned[gpu] == 2 for gpu in gpus)
