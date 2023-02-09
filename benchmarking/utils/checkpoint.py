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
from typing import Callable, Any, Optional, Dict
import argparse
import os

from syne_tune.constants import ST_CHECKPOINT_DIR


def add_checkpointing_to_argparse(parser: argparse.ArgumentParser):
    """
    To be called for the argument parser in the endpoint script.
    Arguments added here are optional. If checkpointing is not supported,
    they are simply not parsed.

    :param parser: Parser to add extra arguments to
    """
    parser.add_argument(f"--{ST_CHECKPOINT_DIR}", type=str)


def resume_from_checkpointed_model(
    config: Dict[str, Any], load_model_fn: Callable[[str], int]
) -> int:
    """
    Checks whether there is a checkpoint to be resumed from. If so, the
    checkpoint is loaded by calling ``load_model_fn``. This function takes
    a local pathname (to which it appends a filename). It returns
    resume_from, the resource value (e.g., epoch) the checkpoint was written
    at. If it fails to load the checkpoint, it may return 0. This skips
    resuming from a checkpoint. This resume_from value is returned.

    If checkpointing is not supported in ``config``, or no checkpoint is
    found, resume_from = 0 is returned.

    :param config: Configuration the training script is called with
    :param load_model_fn: See above, must return ``resume_from``. See
        :func:`pytorch_load_save_functions` for an example
    :return: ``resume_from`` (0 if no checkpoint has been loaded)
    """
    resume_from = 0
    local_path = config.get(ST_CHECKPOINT_DIR)
    if local_path is not None and os.path.exists(local_path):
        resume_from = load_model_fn(local_path)
        trial_id = config.get("trial_id")
        if trial_id is not None:
            print(
                f"Trial {trial_id}: Loading checkpoint [resume_from = "
                f"{resume_from}, local_path = {local_path}]"
            )
    return resume_from


def checkpoint_model_at_rung_level(
    config: Dict[str, Any], save_model_fn: Callable[[str, int], Any], resource: int
):
    """
    If checkpointing is supported, checks whether a checkpoint is to be
    written. This is the case if the checkpoint dir is set in ``config``.
    A checkpoint is written by calling ``save_model_fn``, passing the
    local pathname and resource.

    Note: Why is ``resource`` passed here? In the future, we want to support
    writing checkpoints only for certain resource levels. This is useful if
    writing the checkpoint is expensive compared to the time needed to
    run one resource unit.

    :param config: Configuration the training script is called with
    :param save_model_fn: See above. See :func:`pytorch_load_save_functions` for
        an example
    :param resource: Current resource level (e.g., number of epochs done)
    """
    local_path = config.get(ST_CHECKPOINT_DIR)
    if local_path is not None:
        save_model_fn(local_path, resource)
        trial_id = config.get("trial_id")
        if trial_id is not None:
            print(
                f"Trial {trial_id}: Saving checkpoint [resource = "
                f"{resource}, local_path = {local_path}]"
            )


RESOURCE_NAME = "st_resource"

STATE_DICT_PREFIX = "st_state_dict_"

MUTABLE_STATE_PREFIX = "st_mutable_"


def pytorch_load_save_functions(
    state_dict_objects: Dict[str, Any],
    mutable_state: Optional[dict] = None,
    fname: str = "checkpoint.json",
):
    """
    Provides default ``load_model_fn``, ``save_model_fn`` functions for standard
    PyTorch models (arguments to :func:`resume_from_checkpointed_model`,
    :func:`checkpoint_model_at_rung_level`).

    :param state_dict_objects: Dict of PyTorch objects implementing ``state_dict``
        and ``load_state_dict``
    :param mutable_state: Optional. Additional dict with elementary value
        types
    :param fname: Name of local file (path is taken from config)
    :return: ``load_model_fn, save_model_fn``
    """
    import torch

    def load_model_fn(local_path: str) -> int:
        _mutable_state, local_filename = _common_init(local_path)
        try:
            checkpoint = torch.load(local_filename)
            resume_from = int(checkpoint[RESOURCE_NAME])
            for k, v in state_dict_objects.items():
                v.load_state_dict(checkpoint[STATE_DICT_PREFIX + k])
            for k in _mutable_state:
                v = checkpoint[MUTABLE_STATE_PREFIX + k]
                v_old = _mutable_state.get(k)
                if v_old is not None:
                    v = type(v_old)(v)
                _mutable_state[k] = v
        except Exception:
            resume_from = 0
        return resume_from

    def save_model_fn(local_path: str, resource: int):
        os.makedirs(local_path, exist_ok=True)
        _mutable_state, local_filename = _common_init(local_path)
        local_filename = os.path.join(local_path, fname)
        checkpoint = {
            STATE_DICT_PREFIX + k: v.state_dict() for k, v in state_dict_objects.items()
        }
        checkpoint[RESOURCE_NAME] = resource
        for k, v in _mutable_state.items():
            checkpoint[MUTABLE_STATE_PREFIX + k] = v
        torch.save(checkpoint, local_filename)

    def _common_init(local_path: str) -> (dict, str):
        if mutable_state is None:
            _mutable_state = dict()
        else:
            _mutable_state = mutable_state
        local_filename = os.path.join(local_path, fname)
        return _mutable_state, local_filename

    return load_model_fn, save_model_fn
