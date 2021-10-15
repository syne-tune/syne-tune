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
import os
import tempfile
import json

from sagemaker_tune.backend.backend import get_backend_type, BACKEND_TYPES, ENV_BACKEND
from sagemaker_tune.report import Reporter, retrieve


def test_report_logger():
    report = Reporter(file_based=False)

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


def test_report_file():
    report = Reporter(file_based=True)
    prevpath = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        report(train_nll=1.45, time=1.0, step=2)
        report(train_nll=1.2, time=2.0, step=3)
        content = os.listdir(tmpdir)
        reports = [f for f in content if f.startswith('report')]
        assert len(reports) == 2
        reports.sort()
        with open(reports[0], 'r') as f:
            report0 = json.load(f)
        assert report0['train_nll'] == 1.45
        assert report0['time'] == 1.0
        assert report0['step'] == 2
        with open(reports[1], 'r') as f:
            report1 = json.load(f)
        assert report1['train_nll'] == 1.2
        assert report1['time'] == 2.0
        assert report1['step'] == 3

    os.chdir(prevpath)


def test_report_env():
    prev_backend = os.getenv(ENV_BACKEND)
    os.environ[ENV_BACKEND] = BACKEND_TYPES['queue']
    assert get_backend_type() == BACKEND_TYPES['queue']

    report = Reporter()
    assert report.file_based

    os.environ[ENV_BACKEND] = BACKEND_TYPES['local']
    report = Reporter()
    assert not report.file_based

    # avoid side effect
    if prev_backend:
        os.environ[ENV_BACKEND] = prev_backend
    else:
        os.environ.pop(ENV_BACKEND)
