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
import hashlib
import json
import logging
import types
from pathlib import Path
from typing import Dict, Callable, Optional

import dill

from syne_tune.backend import LocalBackend
from syne_tune.config_space import to_dict, Domain


def file_md5(filename: str) -> str:
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class PythonBackend(LocalBackend):
    def __init__(
        self,
        tune_function: Callable,
        config_space: Dict[str, object],
        rotate_gpus: bool = True,
        delete_checkpoints: bool = False,
    ):
        """
        A backend that supports the tuning of Python functions (if you rather want to tune an endpoint script such as
        "train.py", then you should rather use `LocalBackend`). The function `tune_function` should be serializable,
        should not reference any global variable or module and should have as arguments all the keys of `config_space`.
        When deserializing, a md5 is checked to ensure consistency.

        For instance, the following function is a valid way of defining a backend on top of simple function:

        ```python
        def f(x):
            import logging
            import time
            from syne_tune import Reporter
            root = logging.getLogger()
            root.setLevel(logging.DEBUG)
            reporter = Reporter()
            for i in range(5):
                reporter(step=i + 1, y=x + i)


        from syne_tune.backend.python_backend.python_backend import PythonBackend
        from syne_tune.config_space import uniform

        config_space = {"x": uniform(-10, 10)}
        backend = PythonBackend(tune_function=f, config_space=config_space)
        ```
        See `examples/launch_height_python_backend.py` for a full example.
        :param tune_function: a python function to be tuned, the function should call Syne Tune reporter to report
        metrics and be serializable, imports should be performed inside the function body.
        :param config_space: config_space used to in Syne Tune, it must corresponds to key words arguments of
        `tune_function`.
        :param rotate_gpus: in case several GPUs are present, each trial is
            scheduled on a different GPU. A new trial is preferentially
            scheduled on a free GPU, and otherwise the GPU with least prior
            assignments is chosen. If False, then all GPUs are used at the same
            time for all trials.
        :param delete_checkpoints: If True, checkpoints of stopped or completed
            trials are deleted
        """
        super(PythonBackend, self).__init__(
            entry_point=str(Path(__file__).parent / "python_entrypoint.py"),
            rotate_gpus=rotate_gpus,
            delete_checkpoints=delete_checkpoints,
        )
        self.config_space = config_space
        # save function without reference to global variables or modules
        self.tune_function = types.FunctionType(tune_function.__code__, {})
        self.tune_function_path = self.local_path / "tune_function"

    def set_path(
        self, results_root: Optional[str] = None, tuner_name: Optional[str] = None
    ):
        super(PythonBackend, self).set_path(
            results_root=results_root, tuner_name=tuner_name
        )
        if self.local_path.exists():
            logging.error(
                f"path {self.local_path} already exists, make sure you have a unique tuner name."
            )
        self.tune_function_path = self.local_path / "tune_function"

    def _schedule(self, trial_id: int, config: Dict):
        if not (self.tune_function_path / "tune_function.dill").exists():
            self.save_tune_function(self.tune_function)
        config = config.copy()
        config["tune_function_root"] = str(self.tune_function_path)
        # to detect if the serialized function is the same as the one passed by the user, we pass the md5 to the
        # endpoint script. The hash is checked before executing the function.
        config["tune_function_hash"] = file_md5(
            self.tune_function_path / "tune_function.dill"
        )
        super(PythonBackend, self)._schedule(trial_id=trial_id, config=config)

    def save_tune_function(self, tune_function):
        self.tune_function_path.mkdir(parents=True, exist_ok=True)
        with open(self.tune_function_path / "tune_function.dill", "wb") as file:
            dill.dump(tune_function, file)
        with open(self.tune_function_path / "configspace.json", "w") as file:
            json.dump(
                {
                    k: to_dict(v) if isinstance(v, Domain) else v
                    for k, v in self.config_space.items()
                },
                file,
            )
