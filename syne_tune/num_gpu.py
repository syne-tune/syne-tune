"""
Adapted from to not run in Shell mode which is unsecure.
https://github.com/aws/sagemaker-rl-container/blob/master/src/vw-serving/src/vw_serving/sagemaker/gpu.py
"""

import logging
import subprocess
import time

_num_gpus = None


def get_num_gpus() -> int:
    """
    Returns the number of available GPUs based on configuration parameters and available hardware GPU devices.
    Gpus are detected by running "nvidia-smi --list-gpus" as a subprocess.
    :return: (int) number of GPUs
    """
    global _num_gpus
    if _num_gpus is None:
        try:
            cmd = "nvidia-smi --list-gpus"
            with open("std.out", "w") as stdout:
                proc = subprocess.Popen(cmd.split(" "), shell=False, stdout=stdout)
            max_trials = 0
            while proc.poll() is None and max_trials < 100:
                time.sleep(0.1)
                max_trials += 1

            if proc.poll() is None:
                raise ValueError("nvidia-smi timed out after 10 secs.")

            if proc.poll() != 0:
                # In cases when nvidia-smi fails, no GPU is available.
                return 0
            else:
                # In cases when nvidia-smi success, we read the number of GPU available
                # communicated by nvidia-smi.
                with open("std.out", "r") as stdout:
                    _num_gpus = len(stdout.readlines())
                return _num_gpus

        except (OSError, FileNotFoundError):
            logging.info(
                "Error launching /usr/bin/nvidia-smi, no GPU could be detected."
            )
            _num_gpus = 0
            return 0
    else:
        return _num_gpus
