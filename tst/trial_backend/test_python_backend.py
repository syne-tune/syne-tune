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
import tempfile
from syne_tune.backend import PythonBackend
from syne_tune.backend.trial_status import Status
from syne_tune.config_space import randint
from tst.util_test import wait_until_all_trials_completed


def f(x):
    import logging
    from syne_tune import Reporter

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    reporter = Reporter()
    for i in range(5):
        reporter(step=i + 1, y=x + i)


def test_python_backend():
    with tempfile.TemporaryDirectory() as local_path:
        import logging

        root = logging.getLogger()
        root.setLevel(logging.INFO)
        backend = PythonBackend(f, config_space={"x": randint(0, 10)})
        backend.set_path(str(local_path))
        backend.start_trial({"x": 2})
        backend.start_trial({"x": 3})

        wait_until_all_trials_completed(backend)

        trials, metrics = backend.fetch_status_results([0, 1])

        for trial, status in trials.values():
            assert status == Status.completed, "\n".join(
                backend.stdout(trial.trial_id)
            ) + "\n".join(backend.stderr(trial.trial_id))

        metrics_first_trial = [metric["y"] for x, metric in metrics if x == 0]
        metrics_second_trial = [metric["y"] for x, metric in metrics if x == 1]
        assert metrics_first_trial == [2, 3, 4, 5, 6]
        assert metrics_second_trial == [3, 4, 5, 6, 7]
