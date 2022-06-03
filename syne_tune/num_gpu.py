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
