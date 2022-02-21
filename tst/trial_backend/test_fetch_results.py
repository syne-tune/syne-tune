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
from typing import List, Dict, Optional

from syne_tune.backend.trial_backend import TrialBackend
from syne_tune.backend.trial_status import TrialResult, Status
from syne_tune.constants import ST_WORKER_TIMESTAMP


class DeterministicBackend(TrialBackend):
    def copy_checkpoint(self, src_trial_id: int, tgt_trial_id: int):
        pass

    # a backend which returns deterministic status and metrics to test the logic of the tuner
    def __init__(self):
        super(DeterministicBackend, self).__init__()
        self.iteration = 0
        self.trialid_to_results = {}
        self.timestamp = 0

    def generate_event(self, trial_id: int, metrics: List[Dict], status: Optional[str] = None):
        for m in metrics:
            m[ST_WORKER_TIMESTAMP] = self.timestamp
            self.timestamp += 1

        if trial_id in self.trialid_to_results:
            old_trial_result = self.trialid_to_results[trial_id]
            old_trial_result.status = status
            old_trial_result.metrics += metrics
        else:
            self.trialid_to_results[trial_id] = TrialResult(
                trial_id=trial_id,
                status=status,
                metrics=metrics,
                config=None,
                creation_time=None,
                training_end_time=None,
            )

    def _all_trial_results(self, trial_ids: List[int]) -> List[TrialResult]:
        return [self.trialid_to_results[trial_id] for trial_id in trial_ids]

    def _resume_trial(self, trial_id: int):
        pass

    def _pause_trial(self, trial_id: int):
        pass

    def _stop_trial(self, trial_id: int):
        pass

    def _schedule(self, trial_id: int, config: Dict):
        pass

    def stdout(self, trial_id: int) -> List[str]:
        return []

    def stderr(self, trial_id: int) -> List[str]:
        return []


def test_dummybackend():
    trial_ids = [3, 7]
    # check that the dummy backend behaves as expected, when we call status 2 observations are created
    backend = DeterministicBackend()
    backend.generate_event(trial_id=3, metrics=[{'metric': 6}])
    backend.generate_event(trial_id=7, metrics=[{'metric': 14}])
    metrics = [trial.metrics for trial in backend._all_trial_results(trial_ids)]
    assert metrics == [[{'metric': 6, ST_WORKER_TIMESTAMP: 0}], [{'metric': 14, ST_WORKER_TIMESTAMP: 1}]]

    backend.generate_event(trial_id=3, metrics=[{'metric': 6}])
    backend.generate_event(trial_id=7, metrics=[{'metric': 14}])
    metrics = [trial.metrics for trial in backend._all_trial_results(trial_ids)]
    assert metrics == [[
        {'metric': 6, ST_WORKER_TIMESTAMP: 0},
        {'metric': 6, ST_WORKER_TIMESTAMP: 2}
    ], [
        {'metric': 14, ST_WORKER_TIMESTAMP: 1},
        {'metric': 14, ST_WORKER_TIMESTAMP: 3}
    ]
    ]


def test_fetch_results_metrics():
    # now check that we only get new metrics when we call fetch_results
    trial_ids = [3, 7]
    backend = DeterministicBackend()
    backend.generate_event(trial_id=3, metrics=[])
    backend.generate_event(trial_id=7, metrics=[])
    _, metrics = backend.fetch_status_results(trial_ids)
    assert metrics == []

    backend.generate_event(trial_id=3, metrics=[{'metric': 6, ST_WORKER_TIMESTAMP: 0}])
    backend.generate_event(trial_id=7, metrics=[{'metric': 14}])
    _, metrics = backend.fetch_status_results(trial_ids)
    assert metrics == [
        (3, {'metric': 6, ST_WORKER_TIMESTAMP: 0}),
        (7, {'metric': 14, ST_WORKER_TIMESTAMP: 1})
    ]

    _, metrics = backend.fetch_status_results(trial_ids)
    assert metrics == []

    backend.generate_event(trial_id=3, metrics=[{'metric': 6}])
    backend.generate_event(trial_id=7, metrics=[{'metric': 14}])
    _, metrics = backend.fetch_status_results(trial_ids)
    assert metrics == [
        (3, {'metric': 6, ST_WORKER_TIMESTAMP: 2}),
        (7, {'metric': 14, ST_WORKER_TIMESTAMP: 3})
    ]

    _, metrics = backend.fetch_status_results(trial_ids)
    assert metrics == []

    for i in range(2):
        backend.generate_event(trial_id=3, metrics=[{'metric': 6}])
        backend.generate_event(trial_id=7, metrics=[{'metric': 14}])

    _, metrics = backend.fetch_status_results(trial_ids)
    assert metrics == [
        (3, {'metric': 6, ST_WORKER_TIMESTAMP: 4}),
        (7, {'metric': 14, ST_WORKER_TIMESTAMP: 5}),
        (3, {'metric': 6, ST_WORKER_TIMESTAMP: 6}),
        (7, {'metric': 14, ST_WORKER_TIMESTAMP: 7})
    ]

    _, metrics = backend.fetch_status_results(trial_ids)
    assert metrics == []


def get_status_metrics(backend, trial_ids):
    trial_status_dict, new_metrics = backend.fetch_status_results(trial_ids)
    trial_statuses = {trial_id: status for (trial_id, (_, status)) in trial_status_dict.items()}
    return trial_statuses, new_metrics


def test_fetch_results_status():
    # now check that we get the expected status when fetching_results
    trial_ids = [3, 7]
    backend = DeterministicBackend()

    backend.generate_event(trial_id=3, metrics=[{'metric': 6}], status=Status.in_progress)
    backend.generate_event(trial_id=7, metrics=[{'metric': 14}], status=Status.in_progress)

    trial_statuses, metrics = get_status_metrics(backend, trial_ids)
    assert metrics == [(3, {'metric': 6, 'st_worker_timestamp': 0}), (7, {'metric': 14, 'st_worker_timestamp': 1})]
    assert trial_statuses == {3: Status.in_progress, 7: Status.in_progress}

    # check that status gets updated to failed
    backend.generate_event(trial_id=7, metrics=[], status=Status.failed)
    trial_statuses, metrics = get_status_metrics(backend, trial_ids)

    assert metrics == []
    assert trial_statuses == {3: Status.in_progress, 7: Status.failed}

    # check that in case the trial completed but metrics are still to be seen, the status is in_progresss
    backend.generate_event(trial_id=3, metrics=[{'metric': 6}, {'metric': 6}], status=Status.completed)
    #     v
    # 6 6 6
    trial_statuses, metrics = get_status_metrics(backend, trial_ids)

    assert metrics == [(3, {'metric': 6, 'st_worker_timestamp': 2}), (3, {'metric': 6, 'st_worker_timestamp': 3})]
    assert trial_statuses == {3: Status.completed, 7: Status.failed}

    #       v
    # 6 6 6
    trial_statuses, metrics = get_status_metrics(backend, trial_ids)

    assert metrics == []
    assert trial_statuses == {3: Status.completed, 7: Status.failed}

    #       v
    # 6 6 6
    trial_statuses, metrics = get_status_metrics(backend, trial_ids)

    assert metrics == []
    assert trial_statuses == {3: Status.completed, 7: Status.failed}