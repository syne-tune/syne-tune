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
from typing import List, Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np

from syne_tune.blackbox_repository.blackbox import Blackbox
from syne_tune.blackbox_repository.serialize import (
    serialize_configspace,
    deserialize_configspace,
    deserialize_metadata,
    serialize_metadata,
)


class BlackboxTabular(Blackbox):
    def __init__(
        self,
        hyperparameters: pd.DataFrame,
        configuration_space: dict,
        fidelity_space: dict,
        objectives_evaluations: np.array,
        fidelity_values: Optional[np.array] = None,
        objectives_names: Optional[List[str]] = None,
    ):
        """
        Blackbox that contains tabular evaluations (e.g. all hyperparameters evaluated on all fidelities).
        We use a separate class as BlackboxOffline as performance improvement can be made by avoiding to repeat
        hyperparameters and by storing all evaluations in a single table.
        :param hyperparameters: dataframe of hyperparameters, shape (num_evals, num_hps), columns must match
        hyperparameter names of configuration_space
        :param configuration_space:
        :param fidelity_space: only single fidelity supported for now
        :param objectives_evaluations: values of recorded objectives, must have shape
        (num_evals, num_seeds, num_fidelities, num_objectives)
        :param fidelity_values: values of the `num_fidelities` fidelities, default to [1, ..., `num_fidelities`]
        :param objectives_names:
        """
        super(BlackboxTabular, self).__init__(
            configuration_space=configuration_space,
            fidelity_space=fidelity_space,
            objectives_names=objectives_names,
        )
        # todo missing-value support, should boils down to droping nans in `hyperparameter_objectives_values`
        num_hps = len(hyperparameters.columns)

        assert objectives_evaluations.ndim == 4
        (
            num_evals,
            num_seeds,
            num_fidelities,
            num_objectives,
        ) = objectives_evaluations.shape

        self.num_seeds = num_seeds
        self.num_fidelities = num_fidelities
        if fidelity_values is None:
            self._fidelity_values = np.arange(num_fidelities) + 1
        else:
            # assert sorted(fidelity_values.tolist()) == fidelity_values
            self._fidelity_values = fidelity_values

        # allows to retrieve the index in the objectives_evaluations of a given fidelity
        self.fidelity_map = {
            value: index for index, value in enumerate(self._fidelity_values)
        }
        self.hyperparameters = hyperparameters

        # builds a dataframe to retrieve in O(1) index given a hyperparameter, we could have use a dict but chose a
        # dataframe instead as 1) it is easier since the hyperparameters are itself given in a dataframe (otherwise
        # we would need to have hashable type from the dataframe value) 2) we can support in the future querying
        # multiple results at once efficiently
        self._hp_cols = list(hyperparameters.columns.values)
        self.hyperparameters_index = hyperparameters.copy()
        self.hyperparameters_index["index"] = hyperparameters.index
        self.hyperparameters_index.set_index(self._hp_cols, inplace=True)

        self.objectives_evaluations = objectives_evaluations
        if objectives_names is None:
            self.objectives_names = [f"y{i}" for i in range(num_objectives)]

        assert len(self.objectives_evaluations) == len(hyperparameters)
        assert len(fidelity_space) == 1, "only support single fidelity for now"
        assert (
            max(self._fidelity_values) <= list(fidelity_space.values())[0].upper
        ), f"{max(self._fidelity_values)}, {max(next(iter(fidelity_space.values())).upper)}"
        assert len(hyperparameters) == len(
            hyperparameters.drop_duplicates()
        ), "some hps are duplicated, use a seed column"
        assert len(configuration_space) == num_hps
        for name in configuration_space.keys():
            assert name in hyperparameters.columns

        assert len(self.objectives_names) == num_objectives

    def _objective_function(
        self,
        configuration: Union[dict, int],
        fidelity: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> dict:
        if seed is not None:
            assert 0 <= seed < self.num_seeds
        else:
            seed = np.random.randint(0, self.num_seeds)
        if not isinstance(configuration, dict):
            objectives_values = self.objectives_evaluations[configuration, seed, :, :]
            return objectives_values
        try:
            key = tuple(configuration[key] for key in self._hp_cols)
            matching_index = self.hyperparameters_index.loc[key].values
        except KeyError:
            raise ValueError(
                f"the hyperparameter {configuration} is not present in available evaluations. Use `add_surrogate(blackbox)` if"
                f" you want to add interpolation or a surrogate model that support querying any configuration."
            )

        df_found = self.hyperparameters.loc[matching_index]
        assert len(df_found) == 1
        index = df_found.index.values[0]

        if fidelity is None:
            # returns all fidelities
            objectives_values = self.objectives_evaluations[index, seed, :, :]
            return objectives_values
        else:
            fidelity_index = self.fidelity_map[list(fidelity.values())[0]]
            objectives_values = self.objectives_evaluations[
                index, seed, fidelity_index, :
            ]
            return dict(zip(self.objectives_names, objectives_values))

    @property
    def fidelity_values(self) -> np.array:
        return self._fidelity_values

    def _impute_objectives_values(self) -> Tuple[pd.DataFrame, np.array]:
        """Replaces nan values in objectives with first previous non-nan value.

        Time objective should be cumulative, otherwise each step will consume additional time.
        """
        # Replace nan with previous value. Assumes that elapsed time is cumulative.
        objectives_evaluations = self.objectives_evaluations.copy()
        hyperparameters = self.hyperparameters.copy()
        (
            num_configs,
            num_seeds,
            num_fidelities,
            num_objectives,
        ) = objectives_evaluations.shape
        for config_idx in range(num_configs):
            for seed_idx in range(num_seeds):
                for fidelity_idx in range(num_fidelities):
                    for objective_idx in range(num_objectives):
                        if np.isnan(
                            objectives_evaluations[config_idx][seed_idx][fidelity_idx][
                                objective_idx
                            ]
                        ):
                            objectives_evaluations[config_idx][seed_idx][fidelity_idx][
                                objective_idx
                            ] = objectives_evaluations[config_idx][seed_idx][
                                fidelity_idx - 1
                            ][
                                objective_idx
                            ]
        # Drop all hyperparameters with all nan objectives.
        nan_mask = np.isnan(objectives_evaluations).any((1, 2, 3))
        hyperparameters = hyperparameters[~nan_mask]
        objectives_evaluations = objectives_evaluations[~nan_mask]
        return hyperparameters, objectives_evaluations

    def hyperparameter_objectives_values(self):
        """
        :return: X, y of shape (num_evals * num_seeds * num_fidelities, num_hps)
        and (num_evals * num_seeds * num_fidelities, num_objectives)
        """
        objectives_evaluations = self.objectives_evaluations
        hyperparameters = self.hyperparameters
        if np.isnan(np.sum(objectives_evaluations)):
            hyperparameters, objectives_evaluations = self._impute_objectives_values()

        Xs = []
        ys = []
        for fidelity_index, fidelity_value in enumerate(self.fidelity_values):
            X = hyperparameters.copy()
            X[list(self.fidelity_space.keys())[0]] = fidelity_value
            for seed in range(self.num_seeds):
                Xs.append(X)
                # (num_evals, num_objectives)
                ys.append(objectives_evaluations[:, seed, fidelity_index, :])
        X = pd.concat(Xs, ignore_index=True)
        y = pd.DataFrame(data=np.vstack(ys), columns=self.objectives_names)
        return X, y

    def rename_objectives(self, objective_name_mapping: Dict[str, str]):
        """
        :param objective_name_mapping: dictionary from old objective name to new one, old objective name must be present
        in the blackbox
        :return: a blackbox with as many objectives as objective_name_mapping
        """
        # todo add test
        for old_name in objective_name_mapping.keys():
            assert old_name in self.objectives_names
        objective_indices = dict(
            zip(self.objectives_names, range(len(self.objectives_names)))
        )
        new_objectives_indices = [
            objective_indices[old_obj_name]
            for old_obj_name in objective_name_mapping.keys()
        ]
        return BlackboxTabular(
            hyperparameters=self.hyperparameters,
            configuration_space=self.configuration_space,
            fidelity_space=self.fidelity_space,
            objectives_evaluations=self.objectives_evaluations[
                :, :, :, new_objectives_indices
            ],
            fidelity_values=self._fidelity_values,
            objectives_names=list(objective_name_mapping.values()),
        )

    def __str__(self):
        (
            num_evals,
            num_seeds,
            num_fidelities,
            num_objectives,
        ) = self.objectives_evaluations.shape
        stats = {
            "total evaluations": self.objectives_evaluations.size // num_fidelities,
            "num fidelities": num_fidelities,
            "evaluated hps": num_evals,
            "seeds": num_seeds,
            "fidelities": num_fidelities,
            "objectives": self.objectives_names,
            "hyperparameter": list(self.configuration_space.keys()),
        }
        stats_str = ", ".join([f"{k}: {v}" for k, v in stats.items()])
        return f"tabular blackbox: {stats_str}"


def serialize(
    bb_dict: Dict[str, BlackboxTabular], path: str, metadata: Optional[dict] = None
):
    # check all blackboxes share the same search space and have evaluated the same hyperparameters
    # pick an arbitrary blackbox
    bb_first = next(iter(bb_dict.values()))
    for bb in bb_dict.values():
        pd.testing.assert_frame_equal(bb.hyperparameters, bb_first.hyperparameters)
        # assert bb.configuration_space == bb_first.configuration_space
        # assert bb.fidelity_space == bb_first.fidelity_space
        assert np.all(bb.fidelity_values == bb_first.fidelity_values)
        assert bb.objectives_names == bb_first.objectives_names
        assert bb.objectives_evaluations.shape == bb_first.objectives_evaluations.shape

    path = Path(path)

    path.mkdir(exist_ok=True)

    serialize_configspace(
        path=path,
        configuration_space=bb_first.configuration_space,
        fidelity_space=bb_first.fidelity_space,
    )

    # we use gzip as snappy is not supported for fastparquet engine compression
    # gzip is slower than the default snappy but more compact
    bb_first.hyperparameters.to_parquet(
        path / "hyperparameters.parquet",
        index=False,
        compression="gzip",
        engine="fastparquet",
    )

    with open(path / "objectives_evaluations.npy", "wb") as f:
        # (num_tasks, num_hps, num_seeds, num_fidelities, num_objectives)
        objectives = np.stack(
            [bb_dict[task].objectives_evaluations for task in bb_dict.keys()]
        )
        np.save(f, objectives.astype(np.float32), allow_pickle=False)

    with open(path / "fidelities_values.npy", "wb") as f:
        np.save(f, bb_first.fidelity_values, allow_pickle=False)

    metadata = metadata.copy() if metadata else {}
    metadata.update(
        {
            "objectives_names": bb_first.objectives_names,
            "task_names": list(bb_dict.keys()),
        }
    )
    serialize_metadata(
        path=path,
        metadata=metadata,
    )


def deserialize(path: str) -> Dict[str, BlackboxTabular]:
    """
    Deserialize blackboxes contained in a path that were saved with `serialize` above.
    TODO: the API is currently dissonant with `serialize`, `deserialize` for BlackboxOffline as `serialize` is there a member.
    A possible way to unify is to have serialize also be a free function for BlackboxOffline.
    :param path: a path that contains blackboxes that were saved with `serialize`
    :return: a dictionary from task name to blackbox
    """
    path = Path(path)

    configuration_space, fidelity_space = deserialize_configspace(path)
    hyperparameters = pd.read_parquet(
        Path(path) / "hyperparameters.parquet", engine="fastparquet"
    )

    metadata = deserialize_metadata(path)
    objectives_names = metadata["objectives_names"]
    task_names = metadata["task_names"]

    with open(path / "fidelities_values.npy", "rb") as f:
        fidelity_values = np.load(f)

    # possibly we could use memmap to avoid memory use or speed-up loading times
    with open(path / "objectives_evaluations.npy", "rb") as f:
        objectives_evaluations = np.load(f)

    return {
        task: BlackboxTabular(
            hyperparameters=hyperparameters,
            configuration_space=configuration_space,
            fidelity_space=fidelity_space,
            objectives_evaluations=objectives_evaluations[i],
            fidelity_values=fidelity_values,
            objectives_names=objectives_names,
        )
        for i, task in enumerate(task_names)
    }
