import os
from pathlib import Path
from typing import Dict

import pytest

from syne_tune.constants import SYNE_TUNE_DEFAULT_FOLDER, SYNE_TUNE_ENV_FOLDER
from syne_tune.util import experiment_path


@pytest.mark.parametrize(
    "tuner_name, local_path, env, expected_path",
    [
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
