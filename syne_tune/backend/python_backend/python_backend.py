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
import logging
import types
from pathlib import Path
from typing import Callable, Dict, Optional, Any

import dill

from syne_tune.backend import LocalBackend
from syne_tune.config_space import config_space_to_json_dict
from syne_tune.util import dump_json_with_numpy


def file_md5(filename: str) -> str:
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class PythonBackend(LocalBackend):
    """
    A backend that supports the tuning of Python functions (if you rather want to
    tune an endpoint script such as "train.py", then you should use
    :class:`LocalBackend`). The function ``tune_function`` should be serializable,
    should not reference any global variable or module and should have as arguments
    a subset of the keys of ``config_space``. When deserializing, a md5 is checked to
    ensure consistency.

    For instance, the following function is a valid way of defining a backend on
    top of a simple function:

    .. code-block:: python

       from syne_tune.backend import PythonBackend
       from syne_tune.config_space import uniform

       def f(x, epochs):
           import logging
           import time
           from syne_tune import Reporter
           root = logging.getLogger()
           root.setLevel(logging.DEBUG)
           reporter = Reporter()
           for i in range(epochs):
               reporter(epoch=i + 1, y=x + i)

       config_space = {
           "x": uniform(-10, 10),
           "epochs": 5,
       }
       backend = PythonBackend(tune_function=f, config_space=config_space)

    See ``examples/launch_height_python_backend.py`` for a complete example.

    Additional arguments on top of parent class
    :class:`~syne_tune.backend.LocalBackend`:

    :param tune_function: Python function to be tuned. The function must call
        Syne Tune reporter to report metrics and be serializable, imports should
        be performed inside the function body.
    :param config_space: Configuration space corresponding to arguments of
        ``tune_function``
    """

    def __init__(
        self,
        tune_function: Callable,
        config_space: Dict[str, object],
        rotate_gpus: bool = True,
        delete_checkpoints: bool = False,
    ):
        super(PythonBackend, self).__init__(
            entry_point=str(Path(__file__).parent / "python_entrypoint.py"),
            rotate_gpus=rotate_gpus,
            delete_checkpoints=delete_checkpoints,
            pass_args_as_json=False,
        )
        self.config_space = config_space
        # save function without reference to global variables or modules
        self.tune_function = types.FunctionType(tune_function.__code__, {})

    @property
    def tune_function_path(self) -> Path:
        return self.local_path / "tune_function"

    def set_path(
        self, results_root: Optional[str] = None, tuner_name: Optional[str] = None
    ):
        super(PythonBackend, self).set_path(
            results_root=results_root, tuner_name=tuner_name
        )
        if self.local_path.exists():
            logging.warning(
                f"Path {self.local_path} already exists, make sure you have a unique tuner name."
            )

    def _schedule(self, trial_id: int, config: Dict[str, Any]):
        if not (self.tune_function_path / "tune_function.dill").exists():
            self.save_tune_function(self.tune_function)
        config = config.copy()
        config["tune_function_root"] = str(self.tune_function_path)
        # to detect if the serialized function is the same as the one passed by the user, we pass the md5 to the
        # endpoint script. The hash is checked before executing the function.
        config["tune_function_hash"] = file_md5(
            str(self.tune_function_path / "tune_function.dill")
        )
        super(PythonBackend, self)._schedule(trial_id=trial_id, config=config)

    def save_tune_function(self, tune_function):
        self.tune_function_path.mkdir(parents=True, exist_ok=True)
        with open(self.tune_function_path / "tune_function.dill", "wb") as file:
            dill.dump(tune_function, file)
        dump_json_with_numpy(
            config_space_to_json_dict(self.config_space),
            filename=self.tune_function_path / "configspace.json",
        )
