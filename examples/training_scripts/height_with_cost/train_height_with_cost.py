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
Derived from `train_height.py`, but add variable cost (elapsed time).
"""
import os
import argparse
import logging
import time
import math

from syne_tune.report import Reporter
from syne_tune.search_space import randint, add_to_argparse
from benchmarking.utils import resume_from_checkpointed_model, \
    checkpoint_model_at_rung_level, add_checkpointing_to_argparse, parse_bool


_config_space = {
    'width': randint(0, 20),
    'height': randint(-100, 100),
}


def height_with_cost_default_params(params=None):
    dont_sleep = str(
        params is not None and params.get('backend') == 'simulated')
    return {
        'max_resource_level': 100,
        'grace_period': 1,
        'reduction_factor': 3,
        'instance_type': 'ml.m5.large',
        'num_workers': 4,
        'framework': 'PyTorch',
        'framework_version': '1.6',
        'dont_sleep': dont_sleep,
    }


def height_with_cost_benchmark(params):
    config_space = dict(
        _config_space,
        epochs=params['max_resource_level'],
        dont_sleep=params['dont_sleep'])
    return {
        'script': __file__,
        'metric': 'mean_loss',
        'mode': 'min',
        'resource_attr': 'epoch',
        'elapsed_time_attr': 'elapsed_time',
        'max_resource_attr': 'epochs',
        'config_space': config_space,
        'supports_simulated': True,
    }


def objective(config):
    dont_sleep = parse_bool(config['dont_sleep'])
    width = config['width']
    height = config['height']

    ts_start = time.time()
    report = Reporter()

    # Checkpointing
    # Since this is a tabular benchmark, checkpointing is not really needed.
    # Still, we use a "checkpoint" file in order to store the epoch at which
    # the evaluation was paused, since this information is not passed

    def load_model_fn(local_path: str) -> int:
        local_filename = os.path.join(local_path, 'checkpoint.json')
        try:
            with open(local_filename, 'r') as f:
                data = json.load(f)
                resume_from = int(data['epoch'])
        except Exception:
            resume_from = 0
        return resume_from

    def save_model_fn(local_path: str, epoch: int):
        os.makedirs(local_path, exist_ok=True)
        local_filename = os.path.join(local_path, 'checkpoint.json')
        with open(local_filename, 'w') as f:
            json.dump({'epoch': str(epoch)}, f)

    resume_from = resume_from_checkpointed_model(config, load_model_fn)

    # Loop over epochs
    cost_epoch = 0.1 + 0.05 * math.sin(width * height)
    elapsed_time_raw = 0
    for epoch in range(resume_from + 1, config['epochs'] + 1):
        mean_loss = 1.0 / (0.1 + width * epoch / 100) + 0.1 * height

        if dont_sleep:
            elapsed_time_raw += cost_epoch
        else:
            time.sleep(cost_epoch)
        elapsed_time = time.time() - ts_start + elapsed_time_raw

        report(
            epoch=epoch,
            mean_loss=mean_loss,
            elapsed_time=elapsed_time)

        # Write checkpoint (optional)
        if epoch == config['epochs']:
            checkpoint_model_at_rung_level(config, save_model_fn, epoch)


if __name__ == '__main__':
    # Benchmark-specific imports are done here, in order to avoid import
    # errors if the dependencies are not installed (such errors should happen
    # only when the code is really called)
    import json

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--dont_sleep', type=str, required=True)
    add_to_argparse(parser, _config_space)
    add_checkpointing_to_argparse(parser)

    args, _ = parser.parse_known_args()

    objective(config=vars(args))
