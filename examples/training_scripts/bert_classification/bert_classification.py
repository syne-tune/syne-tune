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
Script to fine-tune a custom BERT model on a classification task
"""

import logging
import argparse

from sagemaker_tune.report import Reporter
from sagemaker_tune.search_space import randint, choice, loguniform, add_to_argparse, uniform
from sagemaker_tune.constants import SMT_CHECKPOINT_DIR
from benchmarks.checkpoint import add_checkpointing_to_argparse


_config_space = {
    'weight_init_seed': choice([str(i) for i in range(8)]),
    'data_shuffling_seed': choice([str(i) for i in range(4)]),
    'n_top_layers_to_reinit': randint(0, 2),
    'lr_decrease_factor': uniform(0.7, 1.0),
    'lr_warmup_proportion': uniform(0.05, 0.5),
    'learning_rate': loguniform(5e-6, 5e-4),
    'weight_decay': loguniform(1e-4, 1e-1),
}


default_params = {
    'weight_init_seed': '0',
    'data_shuffling_seed': '0',
    'n_top_layers_to_reinit': 0,
    'lr_decrease_factor': 1.0,  # 0.95, # see Sun et al. (2019)
    'lr_warmup_proportion': 0.1,  # see e.g. Devlin et al. (2019), Sun et al. (2019), Mosbach et al. (2021)
    'learning_rate': 2e-5,  # see e.g. Devlin et al. (2019), Sun et al. (2019), Mosbach et al. (2021)
    'weight_decay': 1e-2,  # see e.g. Mosbach et al. (2021)
    # Search over these parameters, freeze others:
    'free_parameters': "weight_init_seed learning_rate weight_decay",
    'dataset_path': './data',
    'dataset_name': 'rte',
    'model_name': 'bert-base-uncased',
    'n_epochs': 6,
    'n_evals_per_epoch': 32,
    'train_batch_size': 14,
    'eval_batch_size': 32,
    'n_train_val_data': -1,
    'n_test_data': -1,
    'train_val_split': 0.9,
    'log_interval': 100,
    'reinit_pooler': 'False',
    'dataset_split_seed': 0,
}


dataset_info = {
    'imdb': {'n_classes': 2, 'eval_metric': 'acc'},
    'rte': {'n_classes': 2, 'eval_metric': 'acc'},
    'mrpc': {'n_classes': 2, 'eval_metric': 'f1'},
}


MAX_RESOURCE_ATTR = 'max_num_evaluations'

RUNG_LEVELS_ATTR = 'rung_levels'


def bert_classification_default_params(params=None):
    # Set instance type depending on the backend
    if params is not None and params.get('backend') == 'sagemaker':
        instance_type = 'ml.g4dn.xlarge'    # 1 GPU
        num_workers = 8
    else:
        instance_type = 'ml.g4dn.12xlarge'  # 4 GPUs
        num_workers = 4

    points_to_evaluate = {name: default_params[name] for name in _config_space}
    return {
        **default_params,
        'grace_period': 16,
        'reduction_factor': 3,
        'max_resource_attr': MAX_RESOURCE_ATTR,
        'instance_type': instance_type,
        'num_workers': num_workers,
        'framework': 'HuggingFace',
        'framework_version': '4.4',
        'pytorch_version': '1.6',
        'py_version': 'py36',
        'points_to_evaluate': points_to_evaluate,
        'rung_levels_attr': RUNG_LEVELS_ATTR,
    }


def bert_classification_benchmark(params):
    _params = {k: v for k, v in params.items() if k in default_params}
    config_space = {**default_params, **_params}
    free_parameters = config_space['free_parameters'].split(' ')
    assert free_parameters, "free_parameters must not be empty"
    for name in free_parameters:
        assert name in _config_space, \
            f"name = '{name}' in free_parameters is not a hyperparameter " +\
            f"name ({_config_space.keys()})"
        config_space[name] = _config_space[name]
    print(f"bert_classification: Free hyperparameters are {free_parameters}")
    # Set maximum resource level. The default is the product of
    # `n_epochs` and `n_evals_per_epoch`. This default is set here, because
    # the user may overrides defaults for these latter two
    max_resource_level = params.get('max_resource_level')
    if max_resource_level is None:
        max_resource_level = params['n_evals_per_epoch'] * params['n_epochs']
    config_space[MAX_RESOURCE_ATTR] = max_resource_level
    eval_metric = dataset_info[config_space['dataset_name']]['eval_metric']

    return {
        'script': __file__,
        'metric': eval_metric,
        'mode': 'max',
        'resource_attr': 'n_evaluations',
        'config_space': config_space,
    }


def objective(config: dict):
    """
    Compute the objective function value, which is the validation accuracy of the model

    Args:
        config: The hyperparameter configuration to evaluate
    """

    print(f"config: {config}")  # DEBUG!
    # Prepare model (set seed for weight initialization)
    print("PREPARING MODEL...")
    set_seed(config['weight_init_seed'])
    bert_model = BERTClassificationModel(
        model_name=config['model_name'],
        n_classes=dataset_info[config['dataset_name']]['n_classes'])

    # Prepare data (pass seed for dataset split)
    print("PREPARING DATA...")
    train_dataset, val_dataset, test_dataset = prepare_data(
        dataset_name=config['dataset_name'],
        tokenizer=bert_model.tokenizer,
        n_train_val_data=config['n_train_val_data'],
        n_test_data=config['n_test_data'],
        dataset_path=config['dataset_path'],
        train_val_split=config['train_val_split'],
        seed=config['dataset_split_seed'],
    )

    # Define data loaders (set seed for mini-batch shuffling;
    # see also https://discuss.pytorch.org/t/random-seed-initialization/7854)
    set_seed(config['data_shuffling_seed'])
    worker_init_fn = lambda s=config['data_shuffling_seed']: set_seed(s)
    train_loader = DataLoader(
        train_dataset, batch_size=config['train_batch_size'], drop_last=True,
        worker_init_fn=worker_init_fn, num_workers=0)
    val_loader = DataLoader(
        val_dataset, batch_size=config['eval_batch_size'], drop_last=True,
        worker_init_fn=worker_init_fn, num_workers=0)
    #test_loader = DataLoader(
    #    test_dataset, batch_size=config['eval_batch_size'], drop_last=True,
    #    worker_init_fn=worker_init_fn, num_workers=0)

    # Do not want to count the time to prepare the dataset
    report = Reporter()

    # Prepare optimizer and learning rate scheduler
    print("PREPARING OPTIMIZER...")
    optimizer, scheduler = prepare_optimizer(
        model=bert_model,
        lr=config['learning_rate'],
        lr_decrease_factor=config['lr_decrease_factor'],
        lr_warmup_proportion=config['lr_warmup_proportion'],
        n_training_steps=len(train_loader) * config['n_epochs'])

    # The scheduler may tell us at which resource levels to evaluate
    rung_levels = config.get(RUNG_LEVELS_ATTR)
    if rung_levels is not None:
        rung_levels = [int(x) for x in rung_levels.split()]
        print(f"Evaluations are only done at resource levels {rung_levels}")

    # Fine-tune model (set weight seed again in case we want to re-init some layers)
    print("TRAINING MODEL...")
    set_seed(config['weight_init_seed'])
    bert_model.fit(
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        eval_loader=val_loader, 
        n_epochs=config['n_epochs'],
        n_evals_per_epoch=config['n_evals_per_epoch'],
        max_num_evaluations = config[MAX_RESOURCE_ATTR],
        report=report,
        log_steps=config['log_interval'] // config['train_batch_size'],
        reinit_pooler=config['reinit_pooler'],
        n_top_layers_to_reinit=config['n_top_layers_to_reinit'],
        eval_metric=dataset_info[config['dataset_name']]['eval_metric'],
        checkpoint_config={
            key: config[key] for key in [SMT_CHECKPOINT_DIR, 'trial_id'] if key in config},
        rung_levels=rung_levels,
    )


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from utils import prepare_data, prepare_optimizer, set_seed
    from bert_classification_model import BERTClassificationModel

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Create argument parser with fixed (i.e. non-tunable) hyperparameters
    parser = argparse.ArgumentParser()
    for param, default in default_params.items():
        if param not in _config_space:
            parser.add_argument(
                '--' + param, type=type(default), default=default)
    parser.add_argument('--' + MAX_RESOURCE_ATTR, type=int, required=True)
    parser.add_argument('--' + RUNG_LEVELS_ATTR, type=str)

    # Add tunable hyperparameters and checkpointing to parsers
    add_to_argparse(parser, _config_space)
    add_checkpointing_to_argparse(parser)
    
    # Parse arguments and convert boolean and None values from strings
    args, _ = parser.parse_known_args()
    for arg, val in vars(args).items():
        if isinstance(val, str):
            if val.lower() == "true":
                vars(args)[arg] = True
            elif val.lower() == "false":
                vars(args)[arg] = False
            elif val.lower() == "none":
                vars(args)[arg] = None

    # Convert random seeds from string to int
    args.weight_init_seed = int(args.weight_init_seed)
    args.data_shuffling_seed = int(args.data_shuffling_seed)
    args.dataset_split_seed = int(args.dataset_split_seed)

    # Set conditional hyperparameter values
    if args.n_top_layers_to_reinit > 0:
        args.reinit_pooler = True

    # Evaluate objective with hyperparameter configuration
    objective(config=vars(args))
