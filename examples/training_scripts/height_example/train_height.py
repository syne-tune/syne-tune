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
Example similar to Raytune, https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/skopt_example.py
"""
import logging
import time

from syne_tune import Reporter
from argparse import ArgumentParser


report = Reporter()


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('--steps', type=int)
    parser.add_argument('--width', type=float)
    parser.add_argument('--height', type=float)
    parser.add_argument('--sleep_time', type=float, default=0.1)

    args, _ = parser.parse_known_args()

    width = args.width
    height = args.height
    for step in range(args.steps):
        dummy_score = (0.1 + width * step / 100) ** (-1) + height * 0.1
        # Feed the score back to Syne Tune.
        report(step=step, mean_loss=dummy_score, epoch=step + 1)
        time.sleep(args.sleep_time)
