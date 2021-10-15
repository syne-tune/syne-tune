"""
Taken from
https://github.com/aws/sagemaker-rl-container/blob/master/src/vw-serving/src/vw_serving/sagemaker/gpu.py
"""

import subprocess
import time
from multiprocessing import TimeoutError
import logging


AUTODETECT_GPU_COUNT = "auto"
_num_gpus = None


def _query_num_gpus():
    """
    Returns the number of GPU devices on the host. Returns 0 if the host has no GPU devices.
    """

    global _num_gpus
    if _num_gpus is None:
        COMMAND = 'nvidia-smi -L 2>/dev/null | grep \'GPU [0-9]\' | wc -l'
        TIMEOUT_SECONDS = 75
        STATUS_POLL_INTERVAL_SECONDS = 0.025

        try:
            # todo remove shell mode
            proc = subprocess.Popen(COMMAND, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, bufsize=1)
        except (OSError, ValueError):
            logging.exception("Error launching /usr/bin/nvidia-smi.")
            return 0

        start_time = time.time()

        # Wait for the process to finish
        exitcode = None
        while exitcode is None and time.time() - start_time < TIMEOUT_SECONDS:
            time.sleep(STATUS_POLL_INTERVAL_SECONDS)
            exitcode = proc.poll()

        # Terminate the process if not finished
        if exitcode is None:
            logging.error("nvidia-smi timed out after %s secs", time.time() - start_time)
            proc.terminate()
            raise TimeoutError

        _num_gpus = int(proc.stdout.readline())
        # logging.info("nvidia-smi took: %s secs to identify %d gpus", time.time() - start_time, _num_gpus)

    return _num_gpus


def get_num_gpus(num_gpus=AUTODETECT_GPU_COUNT, **kwargs) -> int:
    """
    Returns the number of available GPUs based on configuration parameters and available hardware GPU devices.

    :param num_gpus: (int or "auto")
        If set to "auto", the function queries and returns the number of available GPUs.
        If set to an integer value, the function returns the value of min(num_gpus, auto_detected_gpu_count)
        Otherwise raises ValueError.
    :param kwargs: additional configuration parameters, not used
    :return: (int) number of GPUs
    """

    # Shortcut execution if what we want is 0 gpu, i.e. only cpu
    if num_gpus == 0:
        return 0

    try:
        num_available_gpus = _query_num_gpus()
    except TimeoutError:
        if num_gpus == AUTODETECT_GPU_COUNT:
            return 0
        else:
            return num_gpus

    if num_gpus == AUTODETECT_GPU_COUNT:
        return num_available_gpus
    else:
        try:
            num_requested_gpus = int(num_gpus)
        except ValueError:
            raise ValueError(
                "Invalid value '{}' provided for hyperparameter '_num_gpus'. '_num_gpus' must be an integer or 'auto'. "
                "Please set the value of '_num_gpus' hyperparameter to 'auto' or an integer value and try again."
                .format(num_gpus))

        if num_requested_gpus > num_available_gpus:
            logging.warning("Request number of gpus: %d, Number of GPUs found: %d",
                            num_requested_gpus, num_available_gpus)
            return num_available_gpus
        else:
            return num_requested_gpus