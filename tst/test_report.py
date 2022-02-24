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
import logging

from syne_tune import Reporter
from syne_tune.report import retrieve


def test_report_logger():
    report = Reporter()

    logging.getLogger().setLevel(logging.INFO)

    report(train_nll=1.45, time=1.0, step=2)
    report(train_nll=1.2, time=2.0, step=3)

    lines = [
        "[tune-metric]: {\"train_nll\": 1.45, \"time\": 1.0, \"step\": 2}\n",
        "[tune-metric]: {\"train_nll\": 1.2, \"time\": 2.0, \"step\": 3}\n",
    ]
    metrics = retrieve(log_lines=lines)
    print(metrics)
    assert metrics == [{'train_nll': 1.45, 'time': 1.0, 'step': 2}, {'train_nll': 1.2, 'time': 2.0, 'step': 3}]