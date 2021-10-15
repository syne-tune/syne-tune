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
Two-layer MLP trained on Fashion MNIST
"""
import os
import argparse
import logging
import time

from sagemaker_tune.report import Reporter
from sagemaker_tune.search_space import randint, uniform, loguniform, \
    add_to_argparse
from benchmarks.checkpoint import resume_from_checkpointed_model, \
    checkpoint_model_at_rung_level, add_checkpointing_to_argparse, \
    pytorch_load_save_functions
from benchmarks.utils import parse_bool


NUM_UNITS_1 = 'n_units_1'

NUM_UNITS_2 = 'n_units_2'


_config_space = {
    NUM_UNITS_1: randint(4, 1024),
    NUM_UNITS_2: randint(4, 1024),
    'batch_size': randint(8, 128),
    'dropout_1': uniform(0, 0.99),
    'dropout_2': uniform(0, 0.99),
    'learning_rate': loguniform(1e-6, 1),
    'wd': loguniform(1e-8, 1)
}


def mlp_fashionmnist_default_params(params=None):
    return {
        'max_resource_level': 81,
        'grace_period': 1,
        'reduction_factor': 3,
        'max_resource_attr': 'epochs',
        'instance_type': 'ml.c5.4xlarge',
        'num_workers': 4,
        'framework': 'PyTorch',
        'framework_version': '1.6',
        'dataset_path': './',
        'report_current_best': 'False',
    }


def mlp_fashionmnist_benchmark(params):
    config_space = dict(
        _config_space,
        dataset_path=params['dataset_path'],
        epochs=params['max_resource_level'],
        report_current_best=params['report_current_best'])
    return {
        'script': __file__,
        'metric': 'accuracy',
        'mode': 'max',
        'resource_attr': 'epoch',
        'elapsed_time_attr': 'elapsed_time',
        'map_reward': '1_minus_x',
        'config_space': config_space,
        'cost_model': get_cost_model(params),
    }


def get_cost_model(params):
    try:
        from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.models.cost.linear_cost_model \
            import FixedLayersMLPCostModel

        num_inputs = 28 * 28
        num_outputs = 10
        num_units_keys = [NUM_UNITS_1, NUM_UNITS_2]
        expected_hidden_layer_width, exp_vals = \
            FixedLayersMLPCostModel.get_expected_hidden_layer_width(
                _config_space, num_units_keys)
        return FixedLayersMLPCostModel(
            num_inputs=num_inputs, num_outputs=num_outputs,
            num_units_keys=num_units_keys,
            expected_hidden_layer_width=expected_hidden_layer_width)
    except Exception:
        return None


# Boilerplate for objective

def download_data(config):
    path = os.path.join(config['dataset_path'], 'FashionMNIST')
    os.makedirs(path, exist_ok=True)
    # Lock protection is needed for backends which run multiple worker
    # processes on the same instance
    lock_path = os.path.join(path, 'lock')
    lock = SoftFileLock(lock_path)
    try:
        with lock.acquire(timeout=120, poll_intervall=1):
            data_train = datasets.FashionMNIST(
                root=path, train=True, download=True,
                transform=transforms.ToTensor())
    except Timeout:
        print(
            "WARNING: Could not obtain lock for dataset files. Trying anyway...",
            flush=True)
        data_train = datasets.FashionMNIST(
            root=path, train=True, download=True,
            transform=transforms.ToTensor())
    return data_train


def split_data(config, data_train):
    # We use 50000 samples for training and 10000 samples for validation
    indices = list(range(data_train.data.shape[0]))
    train_idx, valid_idx = indices[:50000], indices[50000:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    batch_size = config["batch_size"]
    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, sampler=train_sampler,
        drop_last=True)
    valid_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, sampler=valid_sampler,
        drop_last=True)
    return train_loader, valid_loader


def model_and_optimizer(config):
    n_units_1 = config["n_units_1"]
    n_units_2 = config["n_units_2"]
    dropout_1 = config["dropout_1"]
    dropout_2 = config["dropout_2"]
    learning_rate = config["learning_rate"]
    wd = config["wd"]
    # Define the network architecture
    comp_list = [
        nn.Linear(28 * 28, n_units_1),
        nn.Dropout(p=dropout_1),
        nn.ReLU(),
        nn.Linear(n_units_1, n_units_2),
        nn.Dropout(p=dropout_2),
        nn.ReLU(),
        nn.Linear(n_units_2, 10)]
    model = nn.Sequential(*comp_list)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    return {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion}


def train_model(config, state, train_loader):
    model = state['model']
    optimizer = state['optimizer']
    criterion = state['criterion']
    batch_size = config['batch_size']
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data.view(batch_size, -1))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def validate_model(config, state, valid_loader):
    batch_size = config['batch_size']
    model = state['model']
    model.eval()
    correct = 0
    total = 0
    for data, target in valid_loader:
        output = model(data.view(batch_size, -1))
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    return correct / total  # Validation accuracy


def objective(config):
    trial_id = config.get('trial_id')
    debug_log = trial_id is not None
    report_current_best = parse_bool(config['report_current_best'])

    data_train = download_data(config)

    # Do not want to count the time to download the dataset, which can be
    # substantial the first time
    ts_start = time.time()
    report = Reporter()

    train_loader, valid_loader = split_data(config, data_train)

    state = model_and_optimizer(config)

    # Checkpointing
    load_model_fn, save_model_fn = pytorch_load_save_functions(
        state['model'], state['optimizer'])
    # Resume from checkpoint (optional)
    resume_from = resume_from_checkpointed_model(config, load_model_fn)

    current_best = None
    for epoch in range(resume_from + 1, config['epochs'] + 1):
        train_model(config, state, train_loader)
        accuracy = validate_model(config, state, valid_loader)
        elapsed_time = time.time() - ts_start
        if current_best is None or accuracy > current_best:
            current_best = accuracy
        # Feed the score back to Tune.
        objective = current_best if report_current_best else accuracy
        report(
            epoch=epoch,
            accuracy=objective,
            elapsed_time=elapsed_time
        )
        # Write checkpoint (optional)
        checkpoint_model_at_rung_level(config, save_model_fn, epoch)
        if debug_log:
            print("Trial {}: epoch = {}, accuracy = {:.3f}, elapsed_time = {:.2f}".format(
                trial_id, epoch, objective, elapsed_time), flush=True)


if __name__ == '__main__':
    # Benchmark-specific imports are done here, in order to avoid import
    # errors if the dependencies are not installed (such errors should happen
    # only when the code is really called)
    from filelock import SoftFileLock, Timeout
    import torch
    import torch.nn as nn
    from torch.utils.data.sampler import SubsetRandomSampler
    from torchvision import datasets
    from torchvision import transforms


    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--report_current_best', type=str, default='False')
    parser.add_argument('--trial_id', type=str)
    add_to_argparse(parser, _config_space)
    add_checkpointing_to_argparse(parser)

    args, _ = parser.parse_known_args()

    objective(config=vars(args))
