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

from blackbox_repository import load, Blackbox
from syne_tune.backend.simulator_backend.simulator_backend import SimulatorBackend
from syne_tune.backend.trial_status import Status


class _BlackboxSimulatorBackend(SimulatorBackend):

    def __init__(self, elapsed_time_attr: str):
        """
        Allows to simulate any blackbox from blackbox-repository, can be either a blackbox from a registered
        tabulated benchmark (in this case, you should use `BlackboxRepositoryBackend`) or a blackbox given from custom
        code (in this case, you should use `UserBlackboxBackend`), see `examples/launch_simulated_benchmark.py` for
        an example on how to use.
        :param elapsed_time_attr: name of the metric in the dictionary that indicates runtime, it is required in order
        to run simulations.
        """
        super().__init__(
            # TODO we feed a dummy value for entry_point since they are not required
            entry_point=str(Path(__file__)),
            elapsed_time_attr=elapsed_time_attr,
        )

    @property
    def resource_attr(self):
        return next(iter(self.blackbox.fidelity_space.keys()))

    def config_objectives(self, config: dict) -> List[dict]:
        # returns all the fidelities evaluations of a configuration
        res = []
        objective_values = self.blackbox._objective_function(config)
        for fidelity, value in enumerate(self.blackbox.fidelity_values):
            res_dict = dict(zip(self.blackbox.objectives_names, objective_values[fidelity]))
            res_dict[self.resource_attr] = value
            res.append(res_dict)
        return res

    def _run_job_and_collect_results(
            self, trial_id: int,
            config: Optional[dict] = None) -> (str, List[dict]):
        """
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


class BlackboxRepositoryBackend(_BlackboxSimulatorBackend):

    def __init__(
            self,
            blackbox_name: str,
            elapsed_time_attr: str,
            dataset: Optional[str] = None,
    ):
        """
        Backend for evaluations from the blackbox-repository, name of the blackbox and dataset should be present in the
        repository. See `examples/launch_simulated_benchmark.py` for an example on how to use.
        If you want to add a new dataset, see the section `Adding a new dataset section` of 
        `blackbox_repository/README.md`.
        :param blackbox_name: name of a blackbox, should have been registered in blackbox repository.
        :param elapsed_time_attr: name of the metric in the dictionary that indicates runtime, it is required in order
        to run simulations.
        :param dataset:
        """
        super().__init__(elapsed_time_attr=elapsed_time_attr)
        self.blackbox_name = blackbox_name
        self.dataset = dataset
        self._blackbox = None

    @property
    def blackbox(self) -> Blackbox:
        if self._blackbox is not None:
            if self.dataset is None:
                self._blackbox = load(self.blackbox_name)
            else:
                self._blackbox = load(self.blackbox_name)[self.dataset]
        return self._blackbox


class UserBlackboxBackend(_BlackboxSimulatorBackend):
    def __init__(
            self,
            blackbox: Blackbox,
            elapsed_time_attr: str,
    ):
        """
        Backend to run simulation from a user blackbox.
        :param blackbox: blackbox to be used for simulation, see `examples/launch_simulated_benchmark.py` for an example
        on how to use.
        :param elapsed_time_attr:
        """
        super().__init__(elapsed_time_attr=elapsed_time_attr)
        self.blackbox = blackbox
