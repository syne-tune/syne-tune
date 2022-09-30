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
from typing import Dict, List, Optional, Union

import pandas as pd

from syne_tune.blackbox_repository.blackbox import Blackbox
from syne_tune.blackbox_repository.serialize import (
    serialize_configspace,
    deserialize_configspace,
    serialize_metadata,
    deserialize_metadata,
)


class BlackboxOffline(Blackbox):
    def __init__(
        self,
        df_evaluations: pd.DataFrame,
        configuration_space: dict,
        fidelity_space: Optional[dict] = None,
        objectives_names: Optional[List[str]] = None,
        seed_col: Optional[str] = None,
    ):
        """
        A blackbox obtained given offline evaluations each row of the dataframe should contain one evaluation given a
        fixed configuration, fidelity and seed. The columns must corresponds the provided configuration and fidelity
        space, by default all columns that are prefixed by "metric_" are assumed to be metrics but this can be overrided
        by providing metric columns.
        :param df_evaluations:
        :param configuration_space:
        :param fidelity_space:
        :param objectives_names: names of the metrics, by default consider all metrics prefixed by "metric_" to be metrics
        :param seed_col: optional, can be used when multiple seeds are recorded
        """
        if objectives_names is not None:
            self.metric_cols = objectives_names
            for col in objectives_names:
                assert (
                    col in df_evaluations.columns
                ), f"column {col} from metric columns not found in dataframe"
        else:
            self.metric_cols = [
                col for col in df_evaluations.columns if col.startswith("metric_")
            ]

        super(BlackboxOffline, self).__init__(
            configuration_space=configuration_space,
            fidelity_space=fidelity_space,
            objectives_names=objectives_names,
        )

        hp_names = list(configuration_space.keys())
        self.index_cols = hp_names
        if fidelity_space is not None:
            fidelity_names = list(fidelity_space.keys())
            assert len(set(fidelity_names).intersection(hp_names)) == 0
            self.index_cols += fidelity_names

        self.seed_col = seed_col
        if seed_col is not None:
            assert seed_col not in self.index_cols, f"column {seed_col} duplicated"
            self.index_cols.append(seed_col)

        for col in self.index_cols:
            assert (
                col in df_evaluations.columns
            ), f"column {col} from configuration or fidelity space not found in dataframe"

        self.df = df_evaluations.set_index(self.index_cols)

    def hyperparameter_objectives_values(self):
        columns = self.index_cols
        if self.seed_col is not None:
            columns.remove(self.seed_col)
        X = self.df.reset_index().loc[:, columns]
        y = self.df.loc[:, self.metric_cols]
        return X, y

    def _objective_function(
        self,
        configuration: dict,
        fidelity: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Return the dictionary of objectives for a configuration/fidelity/seed.
        :param configuration:
        :param fidelity:
        :param seed:
        :return:
        """
        # todo: we should check range configuration with configspaces
        # query the configuration in the list of available ones
        key_dict = configuration
        if self.seed_col is not None:
            key_dict[self.seed_col] = seed
        if self.fidelity_space is not None and fidelity is not None:
            key_dict.update(fidelity)
        if self.fidelity_space is not None and fidelity is None:
            keys = tuple(set(self.index_cols) - set(self.fidelity_space.keys()))
        else:
            keys = self.index_cols
        output = self.df.xs(tuple(key_dict[col] for col in keys), level=keys).loc[
            :, self.metric_cols
        ]
        if len(output) == 0:
            raise ValueError(
                f"the hyperparameter {configuration} is not present in available evaluations. Use `add_surrogate(blackbox)` if"
                f" you want to add interpolation or a surrogate model that support querying any configuration."
            )
        if fidelity is not None or self.fidelity_space is None:
            return output.iloc[0].to_dict()
        else:
            # TODO select only the fidelity values in the self.fidelity_space, since it might be the case there are more
            #  values in the dataframe. Then the output tensor has larger number of elements than expected num_fidelities.
            return output.to_numpy()

    def __str__(self):
        stats = {
            "total evaluations": len(self.df),
            "objectives": self.objectives_names,
            "hyperparameters": self.configuration_space.get_hyperparameter_names(),
        }
        stats_str = ", ".join([f"{k}: {v}" for k, v in stats.items()])
        return f"offline blackbox: {stats_str}"


def serialize(
    bb_dict: Dict[str, BlackboxOffline], path: str, categorical_cols: List[str] = []
):
    """
    :param bb_dict:
    :param path:
    :param categorical_cols: optional, allow to retrieve columns as categories, lower drastically the memory
     footprint when few values are present
    :return:
    """
    if isinstance(bb_dict, BlackboxOffline):
        # todo hack that allows to call `serialize(bb)` instead of `serialize({"dummy-task-name": bb})`
        # not sure if we should keep it
        bb_dict = {path.stem, bb_dict}

    # check all blackboxes share the same search space and have evaluated the same hyperparameters
    bb_first = next(iter(bb_dict.values()))
    for bb in bb_dict.values():
        assert bb.configuration_space == bb_first.configuration_space
        assert bb.fidelity_space == bb_first.fidelity_space
        assert bb.objectives_names == bb_first.objectives_names

    path = Path(path)
    path.mkdir(exist_ok=True)

    serialize_configspace(
        path=path,
        configuration_space=bb_first.configuration_space,
        fidelity_space=bb_first.fidelity_space,
    )

    for name, bb in bb_dict.items():
        df = bb.df
        df["task"] = name
        # we use gzip as snappy is not supported for fastparquet engine compression
        # gzip is slower than the default snappy but more compact
        df.reset_index().to_parquet(
            path / f"data-{name}.parquet",
            index=False,
            compression="gzip",
            engine="fastparquet",
        )

    serialize_metadata(
        path=path,
        metadata={
            "objectives_names": bb_first.objectives_names,
            "task_names": list(bb_dict.keys()),
            "seed_col": bb_first.seed_col,
            "categorical_cols": categorical_cols,
        },
    )


def deserialize(path: str) -> Union[Dict[str, BlackboxOffline], BlackboxOffline]:
    """
    :param path: where to find blackbox serialized information (at least data.csv.zip and configspace.json)
    :param groupby_col: separate evaluations into a list of blackbox with different task if the column is provided
    :return: list of blackboxes per task, or single blackbox in the case of a single task
    """
    configuration_space, fidelity_space = deserialize_configspace(path)

    assert (
        configuration_space is not None
    ), f"configspace.json could not be found in {path}"

    metadata = deserialize_metadata(path)
    metric_cols = metadata["objectives_names"]
    seed_col = metadata["seed_col"]
    cat_cols = metadata.get("categorical_cols")  # optional
    task_names = metadata.get("task_names")

    # need to specify columns to have categorical encoding of columns (rather than int or float)
    # this is required as it has a massive effect on memory usage; we use fastparquet for the engine as pyarrow does
    # not handle categorization of int/float columns
    df_tasks = {
        task: pd.read_parquet(
            Path(path) / f"data-{task}.parquet",
            categories=cat_cols,
            engine="fastparquet",
        )
        for task in task_names
    }

    return {
        task: BlackboxOffline(
            df_evaluations=df,
            configuration_space=configuration_space,
            fidelity_space=fidelity_space,
            objectives_names=metric_cols,
            seed_col=seed_col,
        )
        for task, df in df_tasks.items()
    }
