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
import os
from pathlib import Path
from typing import Dict

import pytest

from syne_tune.constants import SYNE_TUNE_DEFAULT_FOLDER, SYNE_TUNE_ENV_FOLDER
from syne_tune.util import experiment_path


@pytest.mark.parametrize(
    "tuner_name, local_path, env, expected_path",
    [
        (
            "my-tuner",
            "/tmp/",
            {"SM_MODEL_DIR": "dummy"},
            "/opt/ml/checkpoints/my-tuner",
        ),
        (None, "/tmp/", {"SM_MODEL_DIR": "dummy"}, "/opt/ml/checkpoints/"),
        ("my-tuner", "/tmp/", {}, "/tmp/my-tuner"),
        (None, "/tmp/", {}, "/tmp/"),
        (
            "my-tuner",
            None,
            {},
            str(Path(f"~/{SYNE_TUNE_DEFAULT_FOLDER}").expanduser() / "my-tuner"),
        ),
        (
            "my-tuner",
            None,
            {},
            str(Path(f"~/{SYNE_TUNE_DEFAULT_FOLDER}/my-tuner").expanduser()),
        ),
        (
            "my-tuner",
            None,
            {SYNE_TUNE_ENV_FOLDER: "/home/foo/bar"},
            "/home/foo/bar/my-tuner",
        ),
    ],
)
def test_experiment_path(
    tuner_name: str, local_path: str, env: Dict, expected_path: str
):
    try:
        env_prev = os.environ.copy()
        os.environ.update(env)
        assert experiment_path(tuner_name=tuner_name, local_path=local_path) == Path(
            expected_path
        )
    finally:
        os.environ = env_prev
