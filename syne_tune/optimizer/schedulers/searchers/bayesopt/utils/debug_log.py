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
from typing import Optional, List
import logging
import numpy as np

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common \
    import Configuration
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.config_ext \
    import ExtendedConfiguration
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state \
    import TuningJobState

logger = logging.getLogger(__name__)

__all__ = ['ConfigCounter',
           'DebugLogPrinter']


def _to_key(
        config: Configuration,
        configspace_ext: Optional[ExtendedConfiguration]) -> \
        (str, Optional[int]):
    if configspace_ext is None:
        resource = None
        config_tpl = tuple(
            v for _, v in sorted(config.items(), key=lambda x: x[0]))
    else:
        attr = configspace_ext.resource_attr_name
        resource = config.get(attr)
        if resource is not None:
            config = config.copy()
            del config[attr]
        config_tpl = configspace_ext.hp_ranges.config_to_tuple(config)
    return config_tpl, resource


class ConfigCounter(object):
    """
    Maps a config (non-extended) to a config IDs 0, 1, 2, ...
    If `use_trial_ids` is True, a config is mapped to a trial_id, as passed to
    `add_config`.

    """
    def __init__(
            self, configspace_ext: Optional[ExtendedConfiguration] = None,
            use_trial_ids: bool = True):
        self.config_counter = 0
        self._config_id = dict()
        self.configspace_ext = configspace_ext
        self.use_trial_ids = use_trial_ids
        self.last_recent_id = None

    def add_config(self, config: Configuration, trial_id: Optional[str] = None):
        config_key, resource = _to_key(config, self.configspace_ext)
        assert resource is None, "add_config: config must not be extended"
        _id = self._config_id.get(config_key)
        assert _id is None, \
            f"Config {config} already has been assigned an ID = {_id}"
        self.last_recent_id = trial_id if self.use_trial_ids \
            else str(self.config_counter)
        self._config_id[config_key] = self.last_recent_id
        self.config_counter += 1

    def config_id(self, config: Configuration) -> Optional[str]:
        config_key, resource = _to_key(config, self.configspace_ext)
        _id = self._config_id.get(config_key)
        if resource is None or _id is None:
            return _id
        else:
            return _id + f":{resource}"

    def get_mutable_state(self) -> dict:
        return {
            'config_counter': self.config_counter,
            'config_id': self._config_id,
            'last_recent_id': self.last_recent_id,
            'use_trial_ids': self.use_trial_ids}

    def set_mutable_state(self, state: dict):
        self.config_counter = state['config_counter']
        self._config_id = state['config_id']
        self.last_recent_id = state['last_recent_id']
        self.use_trial_ids = state['use_trial_ids']


class DebugLogPrinter(object):
    """
    Supports a concise debug log.
    The log is made concise and readable by a few properties:
    - configs are mapped to trial IDs. The latter are either passed to
        `start_get_config` and `set_final_config`, or they are 0, 1, 2, ... as
        configs get returned by `get_config`. For multi-fidelity schedulers,
        extended IDs are of the form "<k>:<r>", k the trial ID, <r> the resource
        parameter. Note that even in this case, configs coming out of
        `get_config` are not extended
    - Information about get_config is displayed in a single block. For that,
        different parts are first collected until the end of get_config

    In some use cases, config's are assigned trial_id's externally. If this is
    not done, we maintain our own unique IDs here.
    In either case, a new config must be assigned an ID. This is done in
    `start_get_config`. If `trial_id` is called for a config not yet assigned,
    None is returned.

    """
    def __init__(
            self, configspace_ext: Optional[ExtendedConfiguration] = None):
        self._configspace_ext = configspace_ext
        # config_counter is created with first `start_get_config`, when we
        # know what use_trial_ids should be
        self.config_counter = None
        self.block_info = dict()
        self.get_config_type = None

    def set_configspace_ext(self, configspace_ext: ExtendedConfiguration):
        assert self.config_counter is None, "config_counter already in use"
        self._configspace_ext = configspace_ext

    def trial_id(self, config: Configuration) -> Optional[str]:
        if self.config_counter is not None:
            return self.config_counter.config_id(config)
        else:
            return None

    def start_get_config(self, gc_type, trial_id: Optional[str] = None):
        assert gc_type in {'random', 'BO'}
        assert self.get_config_type is None, \
            "Block for get_config of type '{}' is currently open".format(
                self.get_config_type)
        use_trial_ids = trial_id is not None
        if self.config_counter is None:
            # Delayed creation
            self.config_counter = ConfigCounter(
                self._configspace_ext, use_trial_ids=use_trial_ids)
        else:
            assert use_trial_ids == self.config_counter.use_trial_ids
        self.get_config_type = gc_type
        _id = trial_id if use_trial_ids else self.config_counter.config_counter
        logger.debug(f"Starting get_config[{gc_type}] for trial_id {_id}")

    def set_final_config(
            self, config: Configuration, trial_id: Optional[str] = None):
        assert self.get_config_type is not None, "No block open right now"
        use_trial_ids = trial_id is not None
        assert use_trial_ids == self.config_counter.use_trial_ids
        self.config_counter.add_config(config, trial_id=trial_id)
        entries = ['{}: {}'.format(k, v) for k, v in config.items()]
        msg = '\n'.join(entries)
        self.block_info['final_config'] = msg

    def _get_trial_ids(self, configs: List[Configuration]) -> \
            List[str]:
        lst = []
        for config in configs:
            id = self.trial_id(config)
            if id is not None:
                lst.append(id)
        return lst

    def set_state(self, state: TuningJobState):
        assert self.get_config_type == 'BO', "Need to be in 'BO' block"
        # It is possible for config's in the state not to be registered with
        # config_counter. These are filtered out here.
        # In the multi-fidelity case, `labeled_configs` must be extended
        # configs
        labeled_configs, _ = state.observed_data_for_metric()
        labeled_str = ', '.join(self._get_trial_ids(labeled_configs))
        pending_str = ', '.join(self._get_trial_ids(state.pending_candidates))
        msg = 'Labeled: ' + labeled_str + '. Pending: ' + pending_str
        self.block_info['state'] = msg

    def set_targets(self, targets: np.ndarray):
        assert self.get_config_type == 'BO', "Need to be in 'BO' block"
        msg = 'Targets: ' + str(targets.reshape((-1,)))
        self.block_info['targets'] = msg

    def set_model_params(self, params: dict):
        assert self.get_config_type == 'BO', "Need to be in 'BO' block"
        msg = 'Model params:' + str(params)
        self.block_info['params'] = msg

    def set_fantasies(self, fantasies: np.ndarray):
        assert self.get_config_type == 'BO', "Need to be in 'BO' block"
        msg = 'Fantasized targets:\n' + str(fantasies)
        self.block_info['fantasies'] = msg

    def set_init_config(
            self, config: Configuration, top_scores: np.ndarray = None):
        assert self.get_config_type == 'BO', "Need to be in 'BO' block"
        entries = ['{}: {}'.format(k, v) for k, v in config.items()]
        msg = "Started BO from (top scorer):\n" + '\n'.join(entries)
        if top_scores is not None:
            msg += ("\nTop score values: " + str(top_scores.reshape((-1,))))
        self.block_info['start_config'] = msg

    def set_num_evaluations(self, num_evals: int):
        assert self.get_config_type == 'BO', "Need to be in 'BO' block"
        self.block_info['num_evals'] = num_evals

    def append_extra(self, extra: str):
        if 'extra' in self.block_info:
            self.block_info['extra'] = '\n'.join(
                [self.block_info['extra'], extra])
        else:
            self.block_info['extra'] = extra

    def write_block(self):
        assert self.get_config_type is not None, "No block open right now"
        info = self.block_info
        _trial_id = self.config_counter.last_recent_id
        if 'num_evals' in info:
            parts = ['[{}: {}] ({} evaluations)'.format(
                _trial_id, self.get_config_type, info['num_evals'])]
        else:
            parts = ['[{}: {}]'.format(_trial_id, self.get_config_type)]
        parts.append(info['final_config'])
        debug_parts = []  # Parts for logger.debug
        if self.get_config_type == 'BO':
            if 'start_config' in info:
                debug_parts.append(info['start_config'])
            # The following 3 should be present!
            for name in ('state', 'targets', 'params'):
                v = info.get(name)
                if v is not None:
                    if v == 'targets':
                        debug_parts.append(v)
                    else:
                        parts.append(v)
                else:
                    logger.info(
                        "debug_log.write_block: '{}' part is missing!".format(
                            name))
            if 'fantasies' in info:
                debug_parts.append(info['fantasies'])
        if 'extra' in info:
            debug_parts.append(info['extra'])
        msg = '\n'.join(parts)
        logger.info(msg)
        msg = '\n'.join(debug_parts)
        logger.debug(msg)
        self.get_config_type = None
        self.block_info = dict()

    def get_mutable_state(self) -> dict:
        if self.config_counter is None:
            return dict()
        else:
            return self.config_counter.get_mutable_state()

    def set_mutable_state(self, state: dict):
        assert self.get_config_type is None, \
            "Block for get_config of type '{}' is currently open".format(
                self.get_config_type)
        if len(state) > 0:
            if self.config_counter is None:
                self.config_counter = ConfigCounter(self._configspace_ext)
            self.config_counter.set_mutable_state(state)
        else:
            self.config_counter = None
        self.block_info = dict()
