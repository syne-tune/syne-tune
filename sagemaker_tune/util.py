import os
import string
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import sagemaker

from sagemaker_tune.constants import SAGEMAKER_TUNE_FOLDER


class RegularCallback:
    def __init__(self, callback, call_seconds_frequency: float):
        """
        Allows to call the callback function at most once every `call_seconds_frequency` seconds.
        :param callback:
        :param call_seconds_frequency:
        """
        self.start = datetime.now()
        self.frequency = call_seconds_frequency
        self.callback = callback

    def __call__(self, *args, **kwargs):
        seconds_since_last_call = (datetime.now() - self.start).seconds
        if seconds_since_last_call > self.frequency:
            self.start = datetime.now()
            self.callback(*args, **kwargs)


def experiment_path(
        tuner_name: Optional[str] = None,
        local_path: Optional[str] = None) -> Path:
    """
    :param tuner_name: name of a tuning experiment
    :param local_path: local path where results should be saved when running
        locally outside of Sagemaker, if not specified, then
        `~/{SAGEMAKER_TUNE_FOLDER}/` is used.
    :return: path where to write logs and results for Sagemaker Tune tuner.

    On Sagemaker, results are written under "/opt/ml/checkpoints/" so that files are persisted
    continuously by Sagemaker.
    """
    is_sagemaker = "SM_MODEL_DIR" in os.environ
    if is_sagemaker:
        # if SM_MODEL_DIR is present in the environment variable, this means that we are running on Sagemaker
        # we use this path to store results as it is persisted by Sagemaker.
        return Path('/opt/ml/checkpoints/')
    else:
        # means we are running on a local machine, we store results in a local path
        if local_path is None:
            local_path = Path(f"~/{SAGEMAKER_TUNE_FOLDER}").expanduser()
        else:
            local_path = Path(local_path)
        if tuner_name is not None:
            local_path = local_path / tuner_name
        return local_path


def s3_sanitize_path_name(path: str) -> str:
    """
    S3 does not allow uppercase letters or underscores.
    """
    return path.lower().replace("_", "-")


def s3_experiment_path(
        s3_bucket: Optional[str] = None, experiment_name: Optional[str] = None,
        tuner_name: Optional[str] = None) -> str:
    """
    Returns S3 path for storing results and checkpoints.

    :param s3_bucket: If not given,, the default bucket for the SageMaker
        session is used
    :param experiment_name: If given, this is used as first directory
    :param tuner_name: If given, this is used as second directory
    :return: S3 path
    """
    if s3_bucket is None:
        s3_bucket = sagemaker.Session().default_bucket()
    s3_path = f"s3://{s3_bucket}/{SAGEMAKER_TUNE_FOLDER}"
    for part in (experiment_name, tuner_name):
        if part is not None:
            s3_path += '/' + part
    return s3_sanitize_path_name(s3_path)


def name_from_base(base: Optional[str], default: str, max_length: int = 63) -> str:
    """Append a timestamp to the provided string.

    This function assures that the total length of the resulting string is
    not longer than the specified max length, trimming the input parameter if
    necessary.

    Args:
        base (str): String used as prefix to generate the unique name.
        default (str): String used in case base is None.
        max_length (int): Maximum length for the resulting string (default: 63).

    Returns:
        str: Input parameter with appended timestamp.
    """
    if base is None:
        base = default
    else:
        base = base.replace("_", "-")

    moment = time.time()
    moment_ms = repr(moment).split(".")[1][:3]
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S-{}".format(moment_ms), time.gmtime(moment))
    trimmed_base = base[: max_length - len(timestamp) - 1]
    return "{}-{}".format(trimmed_base, timestamp)


def random_string(length: int) -> str:
    pool = string.ascii_letters + string.digits
    return ''.join(random.choice(pool) for _ in range(length))


def script_checkpoint_example_path():
    """
    Util to get easily the name of an example file
    :return:
    """
    root = Path(__file__).parent.parent
    path = root / "examples" / "training_scripts" / "checkpoint_example" / "checkpoint_example.py"
    assert path.exists()
    return path


def script_height_example_path():
    """
    Util to get easily the name of an example file
    :return:
    """
    root = Path(__file__).parent.parent
    path = root / "examples" / "training_scripts" / "height_example" / "train_height.py"
    assert path.exists()
    return path
