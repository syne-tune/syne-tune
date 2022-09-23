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
from typing import List
import logging
import numpy as np

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    INTERNAL_METRIC_NAME,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import Configuration
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)

logger = logging.getLogger(__name__)

__all__ = ["DebugLogPrinter"]


def _param_dict_to_str(params: dict) -> str:
    parts = []
    for name, param in params.items():
        if isinstance(param, float):
            parts.append(f"{name}: {param:.4e}")
        else:
            parts.append(f"{name}: {param}")
    return "{" + ", ".join(parts) + "}"


class DebugLogPrinter:
    """
    Supports a concise debug log.
    In particular, information about `get_config` is displayed in a single
    block. For that, different parts are first collected until the end of
    `get_config`.

    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self.get_config_trial_id = None
        self.get_config_type = None
        self.block_info = dict()

    def start_get_config(self, gc_type, trial_id: str):
        assert gc_type in {"random", "BO", "grid"}
        assert trial_id is not None
        assert (
            self.get_config_type is None
        ), "Block for get_config of type '{}' is currently open".format(
            self.get_config_type
        )
        self.get_config_trial_id = trial_id
        self.get_config_type = gc_type
        logger.debug(f"Starting get_config[{gc_type}] for trial_id {trial_id}")

    def set_final_config(self, config: Configuration):
        assert self.get_config_type is not None, "No block open right now"
        entries = ["{}: {}".format(k, v) for k, v in config.items()]
        msg = "\n".join(entries)
        self.block_info["final_config"] = msg

    def _observed_trial_ids(self, state: TuningJobState) -> List[str]:
        trial_ids = []
        for ev in state.trials_evaluations:
            trial_id = ev.trial_id
            metric_entry = ev.metrics.get(INTERNAL_METRIC_NAME)
            if metric_entry is not None:
                if isinstance(metric_entry, dict):
                    for resource in metric_entry.keys():
                        trial_ids.append(trial_id + ":" + resource)
                else:
                    trial_ids.append(trial_id)
        return trial_ids

    def _pending_trial_ids(self, state: TuningJobState) -> List[str]:
        trial_ids = []
        for ev in state.pending_evaluations:
            trial_id = ev.trial_id
            resource = ev.resource
            if resource is None:
                trial_ids.append(trial_id)
            else:
                trial_ids.append(trial_id + f":{resource}")
        return trial_ids

    def set_state(self, state: TuningJobState):
        assert self.get_config_type == "BO", "Need to be in 'BO' block"
        labeled_str = ", ".join(self._observed_trial_ids(state))
        msg = "Labeled: " + labeled_str
        if state.pending_evaluations:
            pending_str = ", ".join(self._pending_trial_ids(state))
            msg += ". Pending: " + pending_str
        self.block_info["state"] = msg

    def set_targets(self, targets: np.ndarray):
        assert self.get_config_type == "BO", "Need to be in 'BO' block"
        msg = "Targets: " + str(targets.reshape((-1,)))
        self.block_info["targets"] = msg

    def set_model_params(self, params: dict):
        assert self.get_config_type == "BO", "Need to be in 'BO' block"
        msg = "Model params: " + _param_dict_to_str(params)
        self.block_info["params"] = msg

    def set_fantasies(self, fantasies: np.ndarray):
        assert self.get_config_type == "BO", "Need to be in 'BO' block"
        msg = "Fantasized targets:\n" + str(fantasies)
        self.block_info["fantasies"] = msg

    def set_init_config(self, config: Configuration, top_scores: np.ndarray = None):
        assert self.get_config_type == "BO", "Need to be in 'BO' block"
        entries = ["{}: {}".format(k, v) for k, v in config.items()]
        msg = "Started BO from (top scorer):\n" + "\n".join(entries)
        if top_scores is not None:
            msg += "\nTop score values: " + str(top_scores.reshape((-1,)))
        self.block_info["start_config"] = msg

    def set_num_evaluations(self, num_evals: int):
        assert self.get_config_type == "BO", "Need to be in 'BO' block"
        self.block_info["num_evals"] = num_evals

    def append_extra(self, extra: str):
        if "extra" in self.block_info:
            self.block_info["extra"] = "\n".join([self.block_info["extra"], extra])
        else:
            self.block_info["extra"] = extra

    def write_block(self):
        assert self.get_config_type is not None, "No block open right now"
        info = self.block_info
        trial_id = self.get_config_trial_id
        if "num_evals" in info:
            parts = [
                "[{}: {}] ({} evaluations)".format(
                    trial_id, self.get_config_type, info["num_evals"]
                )
            ]
        else:
            parts = ["[{}: {}]".format(trial_id, self.get_config_type)]
        parts.append(info["final_config"])
        debug_parts = []  # Parts for logger.debug
        if self.get_config_type == "BO":
            if "start_config" in info:
                debug_parts.append(info["start_config"])
            # The following 3 should be present!
            for name in ("state", "targets", "params"):
                v = info.get(name)
                if v is not None:
                    if name == "targets":
                        debug_parts.append(v)
                    else:
                        parts.append(v)
                else:
                    logger.info(
                        "debug_log.write_block: '{}' part is missing!".format(name)
                    )
            if "fantasies" in info:
                debug_parts.append(info["fantasies"])
        if "extra" in info:
            debug_parts.append(info["extra"])
        msg = "\n".join(parts)
        logger.info(msg)
        msg = "\n".join(debug_parts)
        logger.debug(msg)
        self._reset()
