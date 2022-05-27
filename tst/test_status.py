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
from syne_tune.backend.trial_status import Trial, Status
from syne_tune.tuning_status import TuningStatus, print_best_metric_found


def test_status():
    metric_names = ['NLL', 'time']
    status = TuningStatus(metric_names=metric_names)

    trial0 = Trial(trial_id=0, config={"x": 1.0}, creation_time=None)
    trial1 = Trial(trial_id=1, config={"x": 5.0}, creation_time=None)
    status.update(
        trial_status_dict={
            0: (trial0, Status.in_progress),
            1: (trial1, Status.in_progress),
        },
        new_results=[
            (0, {"NLL": 2.0, "time": 10.0, "debug": "str"}),
            (0, {"NLL": 1.0, "time": 12.0, "debug": "str"}),
            (1, {"NLL": 3.0, "time": 5.0, "debug": "str"}),
        ]
    )
    assert status.overall_metric_statistics.max_metrics
    assert status.num_trials_started == 2
    assert status.overall_metric_statistics.max_metrics == {'NLL': 3.0, 'time': 12.0}
    assert status.overall_metric_statistics.min_metrics == {'NLL': 1.0, 'time': 5.0}
    assert status.overall_metric_statistics.sum_metrics == {'NLL': 6.0, 'time': 27.0}

    assert status.trial_metric_statistics[0].max_metrics == {'NLL': 2.0, 'time': 12.0}
    assert status.trial_metric_statistics[0].min_metrics == {'NLL': 1.0, 'time': 10.0}
    assert status.trial_metric_statistics[0].sum_metrics == {'NLL': 3.0, 'time': 22.0}

    status.update(
        trial_status_dict={
            0: (trial0, Status.in_progress),
        },
        new_results=[
            (0, {"NLL": 0.0, "time": 20.0}),
        ]
    )
    assert status.trial_metric_statistics[0].max_metrics == {'NLL': 2.0, 'time': 20.0}
    assert status.trial_metric_statistics[0].min_metrics == {'NLL': 0.0, 'time': 10.0}
    assert status.trial_metric_statistics[0].sum_metrics == {'NLL': 3.0, 'time': 42.0}
    assert status.trial_metric_statistics[0].last_metrics == {'NLL': 0.0, 'time': 20.0}

    print(str(status))

    best_trialid, best_metric = print_best_metric_found(
        tuning_status=status,
        metric_names=metric_names,
        mode='min',
    )
    assert best_trialid == 0
    assert best_metric == 0.0

    best_trialid, best_metric = print_best_metric_found(
        tuning_status=status,
        metric_names=metric_names,
        mode='max',
    )
    assert best_trialid == 1
    assert best_metric == 3.0


def test_stats_are_not_tracked_for_non_numeric_metrics():
    metric_names = ['metric1', 'metric2']
    status = TuningStatus(metric_names=metric_names)

    trial0 = Trial(trial_id=0, config={"x": 1.0}, creation_time=None)
    status.update(
        trial_status_dict={
            0: (trial0, Status.in_progress),
        },
        new_results=[
            (0, {metric_names[0]: 2.0, metric_names[1]: "str"}),
        ]
    )
    assert status.trial_metric_statistics[0].max_metrics == {metric_names[0]: 2.0}
    assert status.trial_metric_statistics[0].min_metrics == {metric_names[0]: 2.0}
    assert status.trial_metric_statistics[0].sum_metrics == {metric_names[0]: 2.0}
    assert status.trial_metric_statistics[0].last_metrics == {metric_names[0]: 2.0, metric_names[1]: "str"}

    status.update(
        trial_status_dict={
            0: (trial0, Status.in_progress),
        },
        new_results=[
            (0, {metric_names[0]: "str", metric_names[1]: 20}),
        ]
    )

    assert status.trial_metric_statistics[0].max_metrics == {metric_names[0]: 2.0}
    assert status.trial_metric_statistics[0].min_metrics == {metric_names[0]: 2.0}
    assert status.trial_metric_statistics[0].sum_metrics == {metric_names[0]: 2.0}
    assert status.trial_metric_statistics[0].last_metrics == {metric_names[0]: "str", metric_names[1]: 20}
