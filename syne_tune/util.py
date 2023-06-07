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
import json
import os
import re
import string
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Union, Dict, Any, Iterable
from time import perf_counter
from contextlib import contextmanager

import numpy as np

from syne_tune.constants import (
    SYNE_TUNE_DEFAULT_FOLDER,
    SYNE_TUNE_ENV_FOLDER,
    ST_DATETIME_FORMAT,
)
from syne_tune.try_import import try_import_aws_message

try:
    import sagemaker
except ImportError:
    print(try_import_aws_message())


class RegularCallback:
    """
    Allows to call the callback function at most once every ``call_seconds_frequency`` seconds.

    :param callback: Callback object
    :param call_seconds_frequency: Wait time between subsequent calls
    """

    def __init__(self, callback: callable, call_seconds_frequency: float):
        self.time_last_recent_call = datetime.now()
        self.frequency = call_seconds_frequency
        self.callback = callback

    def __call__(self, *args, **kwargs):
        seconds_since_last_call = (datetime.now() - self.time_last_recent_call).seconds
        if seconds_since_last_call > self.frequency:
            self.time_last_recent_call = datetime.now()
            self.callback(*args, **kwargs)


def experiment_path(
    tuner_name: Optional[str] = None, local_path: Optional[str] = None
) -> Path:
    """
    Return the path of an experiment which is used both by :class:`~syne_tune.Tuner`
    and to collect results of experiments.

    :param tuner_name: Name of a tuning experiment
    :param local_path: Local path where results should be saved when running
        locally outside of SageMaker. If not specified, then the environment
        variable ``"SYNETUNE_FOLDER"`` is used if defined otherwise ``~/syne-tune/``
        is used. Defining the environment variable ``"SYNETUNE_FOLDER"`` allows to
        override the default path.
    :return: Path where to write logs and results for Syne Tune tuner. On
        SageMaker, results are written to ``"/opt/ml/checkpoints/"`` so that files
        are persisted continuously to S3 by SageMaker.
    """
    is_sagemaker = "SM_MODEL_DIR" in os.environ
    if is_sagemaker:
        # if SM_MODEL_DIR is present in the environment variable, this means that we are running on Sagemaker
        # we use this path to store results as it is persisted by Sagemaker.
        result_path = Path("/opt/ml/checkpoints")
    else:
        # means we are running on a local machine, we store results in a local path
        if local_path is None:
            if SYNE_TUNE_ENV_FOLDER in os.environ:
                result_path = Path(os.environ[SYNE_TUNE_ENV_FOLDER]).expanduser()
            else:
                result_path = Path(f"~/{SYNE_TUNE_DEFAULT_FOLDER}").expanduser()
        else:
            result_path = Path(local_path)
    if tuner_name is not None:
        result_path = result_path / tuner_name
    return result_path


def s3_experiment_path(
    s3_bucket: Optional[str] = None,
    experiment_name: Optional[str] = None,
    tuner_name: Optional[str] = None,
) -> str:
    """Returns S3 path for storing results and checkpoints.

    :param s3_bucket: If not given, the default bucket for the SageMaker
        session is used
    :param experiment_name: If given, this is used as first directory
    :param tuner_name: If given, this is used as second directory
    :return: S3 path, ending on "/"
    """
    if s3_bucket is None:
        s3_bucket = sagemaker.Session().default_bucket()
    s3_path = f"s3://{s3_bucket}/{SYNE_TUNE_DEFAULT_FOLDER}/"
    for part in (experiment_name, tuner_name):
        if part is not None:
            s3_path += part + "/"
    return s3_path


def check_valid_sagemaker_name(name: str):
    assert re.compile("^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$").match(
        name
    ), f"{name} should consists in alpha-digits possibly separated by character -"


def sanitize_sagemaker_name(name: str) -> str:
    new_name = name.replace("_", "-")
    check_valid_sagemaker_name(new_name)
    return new_name


def name_from_base(base: Optional[str], default: str, max_length: int = 63) -> str:
    """Append a timestamp to the provided string.

    This function assures that the total length of the resulting string is
    not longer than the specified max length, trimming the input parameter if
    necessary.

    :param base: String used as prefix to generate the unique name
    :param default: String used if :code:`base is None`
    :param max_length: Maximum length for the resulting string (default: 63)
    :return: Input parameter with appended timestamp
    """
    if base is None:
        check_valid_sagemaker_name(default)
        base = default
    else:
        check_valid_sagemaker_name(base)

    moment = time.time()
    moment_ms = repr(moment).split(".")[1][:3]
    format = ST_DATETIME_FORMAT + f"-{moment_ms}"
    timestamp = time.strftime(format, time.gmtime(moment))
    trimmed_base = base[: max_length - len(timestamp) - 1]
    return "{}-{}".format(trimmed_base, timestamp)


def random_string(length: int) -> str:
    pool = string.ascii_letters + string.digits
    return "".join(random.choice(pool) for _ in range(length))


def repository_root_path() -> Path:
    """
    :return: Returns path including ``syne_tune``, ``examples``, ``benchmarking``
    """
    return Path(__file__).parent.parent


def script_checkpoint_example_path() -> Path:
    """
    :return: Path of checkpoint example
    """
    path = (
        repository_root_path()
        / "examples"
        / "training_scripts"
        / "checkpoint_example"
        / "checkpoint_example.py"
    )
    assert path.exists()
    return path


def script_height_example_path() -> Path:
    """
    :return: Path of ``train_heigth`` example
    """
    path = (
        repository_root_path()
        / "examples"
        / "training_scripts"
        / "height_example"
        / "train_height.py"
    )
    assert path.exists()
    return path


@contextmanager
def catchtime(name: str) -> float:
    start = perf_counter()
    try:
        print(f"start: {name}")
        yield lambda: perf_counter() - start
    finally:
        print(f"Time for {name}: {perf_counter() - start:.4f} secs")


def is_increasing(lst: List[Union[float, int]]) -> bool:
    """
    :param lst: List of float or int entries
    :return: Is ``lst`` strictly increasing?
    """
    return all(x < y for x, y in zip(lst, lst[1:]))


def is_positive_integer(lst: List[int]) -> bool:
    """
    :param lst: List of int entries
    :return: Are all entries of ``lst`` of type ``int`` and positive?
    """
    return all(x == int(x) and x >= 1 for x in lst)


def dump_json_with_numpy(
    x: dict, filename: Optional[Union[str, Path]] = None
) -> Optional[str]:
    """
    Serializes dictionary ``x`` in JSON, taking into account NumPy specific
    value types such as ``n.p.int64``.

    :param x: Dictionary to serialize or encode
    :param filename: Name of file to store JSON to. Optional. If not given,
        the JSON encoding is returned as string
    :return: If ``filename is None``, JSON encoding is returned
    """

    def np_encoder(obj):
        if isinstance(obj, np.generic):
            return obj.item()

    if filename is None:
        return json.dumps(x, default=np_encoder)
    else:
        with open(filename, "w") as f:
            json.dump(x, f, default=np_encoder)
        return None


def dict_get(params: Dict[str, Any], key: str, default: Any) -> Any:
    """
    Returns ``params[key]`` if this exists and is not None, and ``default`` otherwise.
    Note that this is not the same as ``params.get(key, default)``. Namely, if ``params[key]``
    is equal to None, this would return None, but this method returns ``default``.

    This function is particularly helpful when dealing with a dict returned by
    :class:`argparse.ArgumentParser`. Whenever ``key`` is added as argument to the parser,
    but a value is not provided, this leads to ``params[key] = None``.

    """
    v = params.get(key)
    return default if v is None else v


def recursive_merge(
    a: Dict[str, Any],
    b: Dict[str, Any],
    stop_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Merge dictionaries ``a`` and ``b``, where ``b`` takes precedence. We
    typically use this to modify a dictionary ``a``, so ``b`` is smaller
    than ``a``. Further recursion is stopped on any node with key in
    ``stop_keys``. Use this for dictionary-valued entries not to be merged,
    but to be replaced by what is in ``b``.

    :param a: Dictionary
    :param b: Dictionary (can be empty)
    :param stop_keys: See above, optional
    :return: Merged dictionary
    """
    if b:
        if stop_keys is None:
            stop_keys = []
        result = dict()
        keys_b = set(b.keys())
        for k, va in a.items():
            if k in keys_b:
                keys_b.remove(k)
                vb = b[k]
                stop_recursion = k in stop_keys
                if isinstance(va, dict) and not stop_recursion:
                    assert isinstance(
                        vb, dict
                    ), f"k={k} has dict value in a, but not in b:\n{va}\n{vb}"
                    result[k] = recursive_merge(va, vb)
                else:
                    assert stop_recursion or not isinstance(
                        vb, dict
                    ), f"k={k} has dict value in b, but not in a:\n{va}\n{vb}"
                    result[k] = vb
            else:
                result[k] = va
        result.update({k: b[k] for k in keys_b})
        return result
    else:
        return a


def find_first_of_type(a: Iterable[Any], typ) -> Optional[Any]:
    try:
        return next(x for x in a if isinstance(x, typ))
    except StopIteration:
        return None
