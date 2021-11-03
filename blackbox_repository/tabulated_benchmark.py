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
from pathlib import Path
from typing import List, Optional

from blackbox_repository.blackbox_tabular import BlackboxTabular
from syne_tune.backend.simulator_backend.simulator_backend import SimulatorBackend
from syne_tune.backend.trial_status import Status


class TabulatedBenchmark(SimulatorBackend):

    def __init__(
            self,
            blackbox: BlackboxTabular,
            elapsed_time_attr: str,
    ):
        super().__init__(
            # TODO we feed a dummy value for entry_point since they are not required
            entry_point=str(Path(__file__)),
            elapsed_time_attr=elapsed_time_attr,
        )
        self.blackbox = blackbox
        self.fidelities = sorted(self.blackbox.fidelity_values)
        self.ressource_attr = next(iter(self.blackbox.fidelity_space.keys()))

    def config_objectives(self, config: dict) -> List[dict]:
        # returns all the fidelities evaluations of a configuration
        res = []
        objective_values = self.blackbox._objective_function(config)
        for fidelity, value in enumerate(self.blackbox.fidelity_values):
            res_dict = dict(zip(self.blackbox.objectives_names, objective_values[fidelity]))
            res_dict[self.ressource_attr] = value
            res.append(res_dict)
        return res

    def _run_job_and_collect_results(
            self, trial_id: int,
            config: Optional[dict] = None) -> (str, List[dict]):
        """
        Runs training evaluation script for trial `trial_id`, using the config
        `trial(trial_id).config`. This is a blocking call, we wait for the
        script to finish, then parse all its results and return them.

        :param trial_id:
        :return: (final status, list of all results reported)
        """
        assert trial_id in self._trial_dict, \
            f"Trial with trial_id = {trial_id} not registered with back-end"
        if config is None:
            config = self._trial_dict[trial_id].config

        # Fetch all results for this trial from the table
        all_results = self.config_objectives(config)

        status = Status.completed
        num_already_before = self._last_metric_seen_index[trial_id]
        if num_already_before > 0:
            assert num_already_before <= len(all_results), \
                f"Found {len(all_results)} total results, but have already " + \
                f"processed {num_already_before} before!"
            results = all_results[num_already_before:]
            # Correct `elapsed_time_attr` values
            elapsed_time_offset = all_results[num_already_before - 1][
                self.elapsed_time_attr]
            for result in results:
                result[self.elapsed_time_attr] -= elapsed_time_offset
        else:
            results = all_results
        return status, results
