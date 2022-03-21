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
from typing import Dict
import json
import time

from syne_tune.tuner_callback import TunerCallback
from syne_tune import Tuner
from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.schedulers import FIFOScheduler
from syne_tune.optimizer.schedulers.searchers.gp_fifo_searcher import \
    ModelBasedSearcher


class StoreSearcherStatesCallback(TunerCallback):
    """
    Stores list of searcher states alongside a tuning run. The list
    is extended by a new state whenever the `TuningJobState` has changed
    compared to the last recently added one.

    This callback is useful to create meaningful unit tests, by sampling
    a given searcher alongside a realistic experiment.

    Works only for `ModelBasedSearcher` searchers. For other searchers, nothing
    is stored.

    """
    def __init__(self):
        super().__init__()
        self._states = []
        self._num_observations = None
        self._start_time = time.time()
        self._searcher = None

    def on_tuning_start(self, tuner: Tuner):
        scheduler = tuner.scheduler
        if isinstance(scheduler, FIFOScheduler):
            searcher = scheduler.searcher
            if isinstance(searcher, ModelBasedSearcher):
                self._searcher = searcher

    def on_trial_result(self, trial: Trial, status: str, result: Dict,
                        decision: str):
        if self._searcher is not None:
            state = self._searcher.state_transformer.state
            num_observations = state.num_observed_cases()
            if self._num_observations is None or \
                    num_observations != self._num_observations:
                searcher_state = self._searcher.get_state()
                searcher_state['elapsed_time'] = \
                    time.time() - self._start_time
                searcher_state['num_observations'] = num_observations
                searcher_state['num_configs'] = len(state.candidate_evaluations)
                self._states.append(searcher_state)
                self._num_observations = num_observations

    @property
    def states(self):
        return self._states

    def searcher_state_as_code(self, pos: int, add_info: bool = False):
        assert 0 <= pos < len(self._states)
        searcher_state = self._states[pos]
        lines = []
        if add_info:
            lines.append(f"# elapsed_time = {searcher_state['elapsed_time']}")
            lines.append(f"# num_observations = {searcher_state['num_observations']}")
            lines.append(f"# num_configs = {searcher_state['num_configs']}")
        model_params = searcher_state['model_params']
        lines.append(f"_model_params = '{json.dumps(model_params)}'")
        #lines.append("model_params = json.loads(_model_params)")
        state = searcher_state['state']
        lines.append(f"_state = '{json.dumps(state)}'")
        #lines.append("state = decode_state(enc_state=json.loads(_state), hp_ranges=hp_ranges)")
        return '\n'.join(lines)
