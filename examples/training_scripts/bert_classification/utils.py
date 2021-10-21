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
""" Utility methods """

from typing import Optional
import re

import torch
import torch.nn.functional as F


def prepare_data(dataset_name: str,
        tokenizer,
        dataset_path: str,
        n_train_val_data: int = -1,
        n_test_data: int = -1,
        train_val_split: float = 0.8,
        seed: int = 0):
    """
    Download, subsample and tokenize the data

    Args:
        dataset_name: The name of the dataset to load
        tokenizer: The Hugging Face tokenizer to use
        dataset_path: The path for storing the dataset
        n_train_val_data: The number of training + validation data points to use
        n_test_data: The number of testing data points to use
        train_val_split: The train data fraction to use for validation
        seed: The random seed for shuffling the data

    Returns:
        The dataset for training
        The dataset for validation
        The dataset for testing
    """

    # Check validity of arguments
    assert dataset_name in ['imdb', 'rte', 'mrpc'], f'Unsupported dataset: {dataset_name}!'

    from datasets import load_dataset

    print(f"Loading {dataset_name} dataset...")

    # Download data
    dataset_identifier = ['imdb'] if dataset_name == 'imdb' else ['glue', dataset_name]
    dataset = load_dataset(*dataset_identifier, cache_dir=dataset_path)

    testset_name = 'test' if dataset_name == 'imdb' else 'validation'
    train_val_dataset = dataset['train']
    test_dataset = dataset[testset_name]

    # Subsample data
    if n_train_val_data > 0:
        train_val_dataset = train_val_dataset.shuffle(seed=seed).select(range(n_train_val_data))
    else:
        n_train_val_data = len(train_val_dataset)
    if n_test_data > 0:
        test_dataset = test_dataset.shuffle(seed=seed).select(range(n_test_data))
    else:
        n_test_data = len(test_dataset)

    # Determine number of training points
    n_train_data = int(train_val_split * n_train_val_data)
    print(f"Loaded {dataset_name} dataset with {n_train_data} train, "\
            f"{n_train_val_data - n_train_data} val. and {n_test_data} test data points.")

    # Split into training and validation data
    train_dataset = train_val_dataset.select(range(n_train_data))
    val_dataset = train_val_dataset.select(range(n_train_data, n_train_val_data))

    # Tokenize data
    def tokenize(batch):
        inputs = [batch['text']] if dataset_name == 'imdb' else [batch['sentence1'], batch['sentence2']]
        return tokenizer(*inputs, padding='max_length', truncation=True)

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # Set data format to PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    return train_dataset, val_dataset, test_dataset


def _get_layerwise_parameter_groups(model):
    """ Returns a list of parameter groups: one group for each BERT layer, indexed from the top (1, 2, ...)
        and a group with index 0 for all remaining layers (i.e. embedding, pooling & classification layers). 
        This is useful for assigning different learning rates to different layers. """

    parameter_groups = [{'idx': i, 'names': [], 'params': []} for i in range(model.n_layers + 1)]
    for name, param in model.named_parameters():
        match = re.search('layer\.(\d)', name)
        idx = 0 if match is None else model.n_layers - int(match.group(1))
        parameter_groups[idx]['names'].append(name)
        parameter_groups[idx]['params'].append(param)
    return parameter_groups


def prepare_optimizer(
        model,
        lr: float,
        lr_decrease_factor: float = 1.0,
        use_linear_lr_schedule_with_warmup: bool = True,
        lr_warmup_proportion: float = 0.1,
        n_training_steps: int = -1,
        correct_bias: bool = True):
    """
    Instantiate the optimizer and, if desired, set layer-dependent learning rates

    Args:
        model: The BERT classification model to train
        lr: The base learning rate to use for training
        lr_decrease_factor: The factor for decreasing the learning rate across layers
        use_linear_lr_schedule_with_warmup: Should we use a linear lr schedule with warm-up?
        lr_warmup_proportion: The proportion of the training time where the full lr is reached
        n_training_steps: The total number of training steps/batches
        correct_bias: Should we use bias correction for Adam?

    Returns:
        The corresponding AdamW optimizer and learning rate scheduler
    """

    from transformers import AdamW, get_linear_schedule_with_warmup

    # Check argument validity
    assert 0.0 < lr_decrease_factor <= 1.0, "LR decrease factor must be in (0,1]!"
    assert not (use_linear_lr_schedule_with_warmup and n_training_steps == -1),\
            "Need to specify number of training steps when using a LR schedule!"

    if lr_decrease_factor < 1.0:
        # Get parameter groups for each layer and define corresponding AdamW optimizer
        parameter_groups = _get_layerwise_parameter_groups(model)
        optimizer = AdamW(parameter_groups, lr=lr, correct_bias=correct_bias)

        # Set different learning rates per layer (exponentially decaying from top to bottom)
        for param_group in optimizer.param_groups:
            param_group["lr"] *= (lr_decrease_factor ** param_group["idx"])
            print(f"Using learning rate {param_group['lr']} for layer(s) with index {param_group['idx']}.")

    else:
        # Define AdamW optimizer for all model parameters
        optimizer = AdamW(model.parameters(), lr=lr, correct_bias=correct_bias)
        print(f"Using learning rate {lr} for all layers.")

    scheduler = None
    if use_linear_lr_schedule_with_warmup:
        # Define linear learning rate schedule that first warms-up to lr and then decreases to zero
        print(f"Using linear LR schedule with warm-up over the first {int(100*lr_warmup_proportion)}% steps.")
        scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=int(lr_warmup_proportion * n_training_steps),
                num_training_steps=n_training_steps)

    return optimizer, scheduler


def compute_metrics(predictions, labels, metric_names=['accuracy', 'f1']):
    """
    Compute the specified evaluation metrics

    Args:
        predictions: The predictions made by the model
        labels: The ground truth labels
        metric_names: The names of the metrics to compute

    Returns:
        The evaluation metric values
    """

    from datasets import load_metric

    metrics = {}
    for metric_name in metric_names:
        metric = load_metric(metric_name)
        metrics[metric_name] = metric.compute(predictions=predictions, references=labels)[metric_name]

    return metrics


def clip_grad_norm(optimizer, max_grad_norm):
    """
    Clip the norm of the gradients for all parameters under `optimizer`.

    Args:
        optimizer: The optimizer holding the parameters whose gradients to clip.
        max_grad_norm: The maximum gradient norm to clip to.
    """

    import torch

    for group in optimizer.param_groups:
        torch.nn.utils.clip_grad_norm_(group['params'], max_grad_norm)


def set_seed(seed):
    """ Set the global random seeds for random, numpy and PyTorch """
    
    import torch
    import random
    import numpy as np

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def cross_entropy(predictions, targets, weight=None, temp=1.0):
    # Define cross-entropy function depending on if we have hard or soft labels
    ce_function = F.cross_entropy if len(targets.shape) == 1 else cross_entropy_with_probs
    if temp == 1.0:
        return ce_function(predictions, targets, weight=weight)
    else:
        return temp**2 * ce_function(predictions / temp, targets / temp, weight=weight)


def cross_entropy_with_probs(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Calculate cross-entropy loss when targets are probabilities (floats), not ints.
    PyTorch's F.cross_entropy() method requires integer labels; it does accept
    probabilistic labels. We can, however, simulate such functionality with a for loop,
    calculating the loss contributed by each class and accumulating the results.
    Libraries such as keras do not require this workaround, as methods like
    "categorical_crossentropy" accept float labels natively.
    Note that the method signature is intentionally very similar to F.cross_entropy()
    so that it can be used as a drop-in replacement when target labels are changed from
    from a 1D tensor of ints to a 2D tensor of probabilities.
    Parameters
    ----------
    input
        A [num_points, num_classes] tensor of logits
    target
        A [num_points, num_classes] tensor of probabilistic target labels
    weight
        An optional [num_classes] array of weights to multiply the loss by per class
    reduction
        One of "none", "mean", "sum", indicating whether to return one loss per data
        point, the mean loss, or the sum of losses
    Returns
    -------
    torch.Tensor
        The calculated loss
    Raises
    ------
    ValueError
        If an invalid reduction keyword is submitted
    Source: https://github.com/snorkel-team/snorkel/blob/master/snorkel/classification/loss.py
    """

    assert input.shape == target.shape, "Inputs and targets must have same shape!"

    num_points, num_classes = input.shape
    # Note that t.new_zeros, t.new_full put tensor on same device as t
    cum_losses = input.new_zeros(num_points)
    for y in range(num_classes):
        target_temp = input.new_full((num_points,), y, dtype=torch.long)
        y_loss = F.cross_entropy(input, target_temp, reduction="none")
        if weight is not None:
            y_loss = y_loss * weight[y]
        cum_losses += target[:, y].float() * y_loss

    if reduction == "none":
        return cum_losses
    elif reduction == "mean":
        return cum_losses.mean()
    elif reduction == "sum":
        return cum_losses.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")


def pooling(bert_output):
    """ Apply average pooling to the BERT output embeddings """
    return bert_output.mean(axis=1)


def _reinit_bert_layer(model, layer):
    """ Re-initializes the given layer using the original BERT initialization distribution """

    # Re-initialize layer using the original BERT initialization distribution
    # and ensure that it can be trained by unfreezing it
    for module in layer.modules():
        model.bert_model._init_weights(module)
        for param in module.parameters():
            param.requires_grad = True


def reinit_parameters(model, reinit_pooler, n_top_layers_to_reinit):
    """ Re-initializes the specified layers using the original BERT initialization distribution """

    # Obtain list of layers to re-initialize
    layers_to_reinit = [model.bert_model.pooler] if reinit_pooler else []
    encoder = model.bert_model.transformer if 'distilbert' in model.model_name else model.bert_model.encoder
    layers_to_reinit += encoder.layer[-n_top_layers_to_reinit:] if n_top_layers_to_reinit > 0 else []

    # Re-initialize and unfreeze layers
    for layer in layers_to_reinit:
        _reinit_bert_layer(model, layer)
