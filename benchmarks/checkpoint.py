# Provides helper functions for endowing benchmarks with model checkpointing.
# Model checkpointing is useful for pause/resume schedulers: once a benchmark
# function resumes training, it can start from the checkpoint, and does not
# have to start from scatch.

from typing import Dict, Callable, Any
import argparse
import os

from sagemaker_tune.constants import SMT_CHECKPOINT_DIR

__all__ = ['add_checkpointing_to_argparse',
           'resume_from_checkpointed_model',
           'checkpoint_model_at_rung_level',
           'pytorch_load_save_functions']


def add_checkpointing_to_argparse(parser: argparse.ArgumentParser):
    """
    To be called for the argument parser in the endpoint script.
    Arguments added here are optional. If checkpointing is not supported,
    they are simply not parsed.

    :param parser:
    """
    parser.add_argument(f"--{SMT_CHECKPOINT_DIR}", type=str)


def resume_from_checkpointed_model(
        config: Dict,
        load_model_fn: Callable[[str], int]) -> int:
    """
    Checks whether there is a checkpoint to be resumed from. If so, the
    checkpoint is loaded by calling `load_model_fn`. This function takes
    a local pathname (to which it appends a filename). It returns
    resume_from, the resource value (e.g., epoch) the checkpoint was written
    at. If it fails to load the checkpoint, it may return 0. This skips
    resuming from a checkpoint. This resume_from value is returned.

    If checkpointing is not supported in `config`, or no checkpoint is
    found, resume_from = 0 is returned.

    :param config:
    :param load_model_fn:
    :return: resume_from (0 if no checkpoint has been loaded)
    """
    resume_from = 0
    local_path = config.get(SMT_CHECKPOINT_DIR)
    if local_path is not None and os.path.exists(local_path):
        resume_from = load_model_fn(local_path)
        trial_id = config.get('trial_id')
        if trial_id is not None:
            print(f"Trial {trial_id}: Loading checkpoint [resume_from = "
                  f"{resume_from}, local_path = {local_path}]")
    return resume_from


def checkpoint_model_at_rung_level(
        config: Dict,
        save_model_fn: Callable[[str, int], Any],
        resource: int):
    """
    If checkpointing is supported, checks whether a checkpoint is to be
    written. This is the case if the checkpoint dir is set in `config`.
    A checkpoint is written by calling `save_model_fn`, passing the
    local pathname and resource.

    Note: Why is `resource` passed here? In the future, we want to support
    writing checkpoints only for certain resource levels. This is useful if
    writing the checkpoint is expensive compared to the time needed to
    run one resource unit.

    :param config:
    :param save_model_fn:
    :param resource:
    """
    local_path = config.get(SMT_CHECKPOINT_DIR)
    if local_path is not None:
        save_model_fn(local_path, resource)
        trial_id = config.get('trial_id')
        if trial_id is not None:
            print(f"Trial {trial_id}: Saving checkpoint [resource = "
                  f"{resource}, local_path = {local_path}]")


def pytorch_load_save_functions(
        model, optimizer, lr_scheduler=None, fname='checkpoint.json'):
    """
    Provides default `load_model_fn`, `save_model_fn` functions for standard
    PyTorch models (arguments to `resume_from_checkpointed_model`,
    `checkpoint_model_at_rung_level`.

    :param model: Pytorch model
    :param optimizer: PyTorch optimizer
    :param lr_scheduler: PyTorch LR scheduler (optional)
    :param fname: Name of local file (path is taken from config)
    :return: load_model_fn, save_model_fn
    """
    import torch

    def load_model_fn(local_path: str) -> int:
        local_filename = os.path.join(local_path, fname)
        try:
            checkpoint = torch.load(local_filename)
            resume_from = int(checkpoint['epoch'])
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(
                    checkpoint['scheduler_state_dict'])
        except Exception:
            resume_from = 0
        return resume_from

    def save_model_fn(local_path: str, epoch: int):
        os.makedirs(local_path, exist_ok=True)
        local_filename = os.path.join(local_path, fname)
        data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}
        if lr_scheduler is not None:
            data['scheduler_state_dict'] = lr_scheduler.state_dict()
        torch.save(data, local_filename)

    return load_model_fn, save_model_fn
