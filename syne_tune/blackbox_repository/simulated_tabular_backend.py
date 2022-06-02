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
from typing import List, Optional, Dict
import logging
import numpy as np

from syne_tune.blackbox_repository import load, add_surrogate
from syne_tune.blackbox_repository.blackbox import Blackbox
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular
from syne_tune.blackbox_repository.utils import metrics_for_configuration

from syne_tune.backend.simulator_backend.simulator_backend import SimulatorBackend
from syne_tune.backend.trial_status import Status
from syne_tune.config_space import to_dict, from_dict, Domain

logger = logging.getLogger(__name__)


class _BlackboxSimulatorBackend(SimulatorBackend):
    def __init__(
        self,
        elapsed_time_attr: str,
        time_this_resource_attr: Optional[str] = None,
        max_resource_attr: Optional[str] = None,
        seed: Optional[int] = None,
        **simulatorbackend_kwargs,
    ):
        """
        Allows to simulate any blackbox from blackbox-repository, can be either a blackbox from a registered
        tabulated benchmark (in this case, you should use `BlackboxRepositoryBackend`) or a blackbox given from custom
        code (in this case, you should use `UserBlackboxBackend`), see `examples/launch_simulated_benchmark.py` for
        an example on how to use.

        In each result reported to the simulator back-end, the value for key
        `elapsed_time_attr` must be the time since since the start of the
        evaluation. For example, if resource (or fidelity) equates to epochs
        trained, this would be the time from start of training until the end
        of the epoch. If the blackbox contains this information in a column,
        `elapsed_time_attr` should be its key, and `time_this_resource_attr`
        should be ignored.

        Some blackboxes only maintain the time required for the current resource,
        counting from the one before. In the example, this would be the time
        spent for the current epoch only. In this case, specify the
        corresponding column name in `time_this_resource_attr`. The corresponding
        values of `elapsed_time_attr` will then be generated as cumulative sums
        and appended to results when passing them to the simulator back-end.

        ATTENTION: If the blackbox maintains cumulative time (elapsed_time),
        this is different from what :class:`SimulatorBackend` requires for
        `elapsed_time_attr`, if a pause and resume scheduler is used.
        Namely, the back-end requires the time since the start of the last recent
        resume. This conversion is done here internally in
        `_run_job_and_collect_results`, which is called for each resume. This
        means that the field `elapsed_time_attr` is not what is received from
        the blackbox table, but instead what the back-end needs.

        `max_resource_attr` plays the same role as in :class:`HyperbandScheduler`.
        If given, it is the key in a configuration `config` for the maximum
        resource. This is used by schedulers which limit each evaluation by
        setting this argument (e.g., promotion-based Hyperband).

        If `seed` is given, entries of the blackbox are queried for this
        seed. Otherwise, a seed is drawn at random for every trial, but the
        same seed is used for all `_run_job_and_collect_results` calls for the
        same trial. This is important for pause and resume scheduling.

        :param elapsed_time_attr: See above
        :param time_this_resource_attr: See above
        :param max_resource_attr: See above
        :param seed: See above
        """
        super().__init__(
            # TODO we feed a dummy value for entry_point since they are not required
            entry_point=str(Path(__file__)),
            elapsed_time_attr=elapsed_time_attr,
            **simulatorbackend_kwargs,
        )
        self._time_this_resource_attr = time_this_resource_attr
        self._max_resource_attr = max_resource_attr
        self.simulatorbackend_kwargs = simulatorbackend_kwargs
        self._seed = seed
        self._seed_for_trial = dict()

    @property
    def blackbox(self) -> Blackbox:
        raise NotImplementedError()

    @property
    def resource_attr(self):
        return next(iter(self.blackbox.fidelity_space.keys()))

    def config_objectives(self, config: dict, seed: int) -> List[dict]:
        if self._max_resource_attr is not None and self._max_resource_attr in config:
            max_resource = int(config[self._max_resource_attr])
            fidelity_range = (min(self.blackbox.fidelity_values), max_resource)
        else:
            fidelity_range = None  # All fidelity values
        return metrics_for_configuration(
            blackbox=self.blackbox,
            config=config,
            resource_attr=self.resource_attr,
            fidelity_range=fidelity_range,
            seed=seed,
        )

    def _run_job_and_collect_results(
        self, trial_id: int, config: Optional[dict] = None
    ) -> (str, List[dict]):
        """
        :param trial_id:
        :return: (final status, list of all results reported)
        """
        assert (
            trial_id in self._trial_dict
        ), f"Trial with trial_id = {trial_id} not registered with back-end"
        if config is None:
            config = self._trial_dict[trial_id].config

        # Seed for query to blackbox. It is important to use the same
        # seed for all queries for the same `trial_id`
        seed = None
        if self._seed is not None:
            seed = self._seed
        elif isinstance(self.blackbox, BlackboxTabular):
            seed = self._seed_for_trial.get(trial_id)
            if seed is None:
                seed = np.random.randint(0, self.blackbox.num_seeds)
                self._seed_for_trial[trial_id] = seed

        # Fetch all results for this trial from the table
        all_results = self.config_objectives(config, seed=seed)
        # Compute and append `elapsed_time_attr` if not provided
        if self._time_this_resource_attr is not None:
            cumulative_sum = 0
            for result in all_results:
                cumulative_sum += result[self._time_this_resource_attr]
                result[self.elapsed_time_attr] = cumulative_sum

        status = Status.completed
        num_already_before = self._last_metric_seen_index[trial_id]
        if num_already_before > 0:
            assert num_already_before <= len(all_results), (
                f"Found {len(all_results)} total results, but have already "
                + f"processed {num_already_before} before!"
            )
            results = all_results[num_already_before:]
            # Correct `elapsed_time_attr` values
            elapsed_time_offset = all_results[num_already_before - 1][
                self.elapsed_time_attr
            ]
            for result in results:
                result[self.elapsed_time_attr] -= elapsed_time_offset
        else:
            results = all_results

        # Makes sure that time is monotonically increasing which may not be the case due to numerical errors or due to
        # the use of a surrogate
        for i in range(1, len(results)):
            results[i][self.elapsed_time_attr] = max(
                results[i][self.elapsed_time_attr],
                results[i - 1][self.elapsed_time_attr] + 0.001,
            )

        return status, results


def make_surrogate(
    surrogate: Optional[str] = None, surrogate_kwargs: Optional[Dict] = None
):
    """
    :param surrogate: optionally, a model that is fitted to predict objectives given any configuration.
    Possible examples: "KNeighborsRegressor" or "MLPRegressor" or "XGBRegressor" which would enable using
    the corresponding Scikit-learn estimator.
    The model is fit on top of pipeline that applies basic feature-processing to convert hyperparameters
    rows in X to vectors. The configuration_space hyperparameters types are used to deduce the types of columns in
     X (for instance CategoricalHyperparameter are one-hot encoded).
    :param surrogate_kwargs: arguments for the scikit-learn estimator, for instance {"n_neighbors": 1} can be used
    if `surrogate="KNeighborsRegressor"` is chosen.
    :return:
    """
    if surrogate is None:
        return None
    else:
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.ensemble import RandomForestRegressor
        import xgboost

        surrogate_dict = {
            "KNeighborsRegressor": KNeighborsRegressor,
            "MLPRegressor": MLPRegressor,
            "XGBRegressor": xgboost.XGBRegressor,
            "RandomForestRegressor": RandomForestRegressor,
        }
        assert surrogate in surrogate_dict, (
            f"surrogate passed {surrogate} is not supported, "
            f"only {list(surrogate_dict.keys())} are available"
        )
        if surrogate_kwargs is None:
            surrogate_kwargs = {}
        return surrogate_dict[surrogate](**surrogate_kwargs)


class BlackboxRepositoryBackend(_BlackboxSimulatorBackend):
    def __init__(
        self,
        blackbox_name: str,
        elapsed_time_attr: str,
        time_this_resource_attr: Optional[str] = None,
        max_resource_attr: Optional[str] = None,
        seed: Optional[int] = None,
        dataset: Optional[str] = None,
        surrogate: Optional[str] = None,
        surrogate_kwargs: Optional[Dict] = None,
        config_space_surrogate: Optional[Dict] = None,
        **simulatorbackend_kwargs,
    ):
        """
        Backend for evaluations from the blackbox-repository, name of the blackbox and dataset should be present in the
        repository. See `examples/launch_simulated_benchmark.py` for an example on how to use.
        If you want to add a new dataset, see the section `Adding a new dataset section` of
        `blackbox_repository/README.md`.

        :param blackbox_name: name of a blackbox, should have been registered in blackbox repository.
        :param elapsed_time_attr:
        :param time_this_resource_attr:
        :param max_resource_attr:
        :param dataset: Selects different versions of the blackbox
        :param surrogate: optionally, a model that is fitted to predict objectives given any configuration.
        Possible examples: "KNeighborsRegressor" or "MLPRegressor" or "XGBRegressor" which would enable using
        the corresponding Scikit-learn estimator.
        The model is fit on top of pipeline that applies basic feature-processing to convert hyperparameters
        rows in X to vectors. The configuration_space hyperparameters types are used to deduce the types of columns in
         X (for instance CategoricalHyperparameter are one-hot encoded).
        :param surrogate_kwargs: arguments for the scikit-learn estimator, for instance {"n_neighbors": 1} can be used
        if `surrogate="KNeighborsRegressor"` is chosen.
        :param config_space_surrogate: if `surrogate` is given, this is the
            configuration space for the surrogate blackbox. If not given, the
            space of the original blackbox is used. However, if this is a tabular
            blackbox, its numerical parameters have categorical domains, which is
            usually not what we want for a surrogate.
        """
        assert (
            config_space_surrogate is None or surrogate is not None
        ), "config_space_surrogate only together with surrogate"
        super().__init__(
            elapsed_time_attr=elapsed_time_attr,
            time_this_resource_attr=time_this_resource_attr,
            max_resource_attr=max_resource_attr,
            seed=seed,
            **simulatorbackend_kwargs,
        )
        self.blackbox_name = blackbox_name
        self.dataset = dataset
        self._blackbox = None
        if surrogate is not None:
            # makes sure the surrogate can be constructed
            make_surrogate(surrogate=surrogate, surrogate_kwargs=surrogate_kwargs)
        self._surrogate = surrogate
        self._surrogate_kwargs = (
            surrogate_kwargs if surrogate_kwargs is not None else {}
        )
        if config_space_surrogate is not None:
            self._config_space_surrogate = {
                k: v for k, v in config_space_surrogate.items() if isinstance(v, Domain)
            }
        else:
            self._config_space_surrogate = None

    @property
    def blackbox(self) -> Blackbox:
        if self._blackbox is None:
            if self.dataset is None:
                self._blackbox = load(self.blackbox_name)
                # TODO: This could fail earlier
                assert not isinstance(self._blackbox, dict), (
                    f"blackbox_name = '{self.blackbox_name}' maps to a dict, "
                    + "dataset argument must be given"
                )
            else:
                self._blackbox = load(self.blackbox_name)[self.dataset]
            if self._surrogate is not None:
                surrogate = make_surrogate(
                    surrogate=self._surrogate, surrogate_kwargs=self._surrogate_kwargs
                )
                self._blackbox = add_surrogate(
                    blackbox=self._blackbox,
                    surrogate=surrogate,
                    configuration_space=self._config_space_surrogate,
                )

        return self._blackbox

    def __getstate__(self):
        # we serialize only required metadata information since the blackbox data is contained in the repository and
        # its raw data does not need to be saved.
        state = {
            "elapsed_time_attr": self.elapsed_time_attr,
            "time_this_resource_attr": self._time_this_resource_attr,
            "max_resource_attr": self._max_resource_attr,
            "seed": self._seed,
            "seed_for_trial": self._seed_for_trial,
            "simulatorbackend_kwargs": self.simulatorbackend_kwargs,
            "blackbox_name": self.blackbox_name,
            "dataset": self.dataset,
            "surrogate": self._surrogate,
            "surrogate_kwargs": self._surrogate_kwargs,
        }
        if self._config_space_surrogate is not None:
            state["config_space_surrogate"] = {
                k: to_dict(v) for k, v in self._config_space_surrogate.items()
            }
        return state

    def __setstate__(self, state):
        super().__init__(
            elapsed_time_attr=state["elapsed_time_attr"],
            time_this_resource_attr=state["time_this_resource_attr"],
            max_resource_attr=state["max_resource_attr"],
            seed=state["seed"],
            **state["simulatorbackend_kwargs"],
        )
        self._seed_for_trial = state["seed_for_trial"]
        self.blackbox_name = state["blackbox_name"]
        self.dataset = state["dataset"]
        self._surrogate = state["surrogate"]
        self._surrogate_kwargs = state["surrogate_kwargs"]
        self._blackbox = None
        if "config_space_surrogate" in state:
            self._config_space_surrogate = {
                k: from_dict(v) for k, v in state["config_space_surrogate"].items()
            }
        else:
            self._config_space_surrogate = None


class UserBlackboxBackend(_BlackboxSimulatorBackend):
    def __init__(
        self,
        blackbox: Blackbox,
        elapsed_time_attr: str,
        time_this_resource_attr: Optional[str] = None,
        max_resource_attr: Optional[str] = None,
        seed: Optional[int] = None,
        **simulatorbackend_kwargs,
    ):
        """
        Backend to run simulation from a user blackbox.

        :param blackbox: blackbox to be used for simulation, see `examples/launch_simulated_benchmark.py` for an example
            on how to use.
        :param elapsed_time_attr:
        :param time_this_resource_attr:
        :param max_resource_attr:
        """
        super().__init__(
            elapsed_time_attr=elapsed_time_attr,
            time_this_resource_attr=time_this_resource_attr,
            max_resource_attr=max_resource_attr,
            seed=seed,
            **simulatorbackend_kwargs,
        )
        self._blackbox = blackbox

    @property
    def blackbox(self) -> Blackbox:
        return self._blackbox
