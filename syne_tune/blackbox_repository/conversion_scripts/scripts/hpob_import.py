import zipfile
import urllib.request
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular
from syne_tune.blackbox_repository.conversion_scripts.scripts import metric_elapsed_time
from syne_tune.blackbox_repository.conversion_scripts.utils import (
    repository_path,
)
from syne_tune.blackbox_repository.conversion_scripts.blackbox_recipe import (
    BlackboxRecipe,
)
from syne_tune.config_space import (
    uniform,
    randint,
    choice,
    config_space_to_json_dict,
    config_space_from_json_dict,
)
from syne_tune.util import catchtime, dump_json_with_numpy

from syne_tune.blackbox_repository.serialize import (
    deserialize_configspace,
    deserialize_metadata,
    serialize_configspace,
    serialize_metadata,
)

BLACKBOX_NAME = "hpob_"
METRIC_ELAPSED_TIME = "metric_elapsed_time"

# configuration_space values taken from "https://raw.githubusercontent.com/machinelearningnuremberg/HPO-B/refs/heads/main/hpob-data/meta-dataset-descriptors.json"

SEARCH_SPACE_4796 = {
    "name": "4796",
    "positions_hps": {"minsplit": 0, "minbucket": 1, "cp": 2},
    "positions_categorical": {},
    "config_space": {
        "minsplit": uniform(0.0, 1.0),
        "minbucket": uniform(0.0, 1.0),
        "cp": uniform(0.0, 1.0),
    },
}

SEARCH_SPACE_5527 = {
    "name": "5527",
    "positions_hps": {"cost": 0, "gamma": 1, "degree": 3},
    "positions_categorical": {"kernel": [4, 5, 6]},
    "config_space": {
        "cost": uniform(0.0, 1.0),
        "gamma": uniform(0.0, 1.0),
        "degree": uniform(0.0, 1.0),
        # "kernel": choice(["_INVALID", "_linear", "_polynomial"]),
        "kernel": choice([0.0, 1.0, 2.0]),
    },
}
SEARCH_SPACE_5636 = {
    "name": "5636",
    "positions_hps": {"minsplit": 0, "cp": 2, "maxdepth": 3, "minbucket": 4},
    "positions_categorical": {},
    "config_space": {
        "minsplit": uniform(0.0, 1.0),
        "minbucket": uniform(0.0, 1.0),
        "cp": uniform(0.0, 1.0),
        "maxdepth": uniform(0.0, 1.0),
    },
}
SEARCH_SPACE_5859 = {
    "name": "5859",
    "positions_hps": {"minsplit": 0, "cp": 2, "maxdepth": 3, "minbucket": 4},
    "positions_categorical": {},
    "config_space": {
        "minsplit": uniform(0.0, 1.0),
        "minbucket": uniform(1.0, 1.0),
        "cp": uniform(0.0, 1.0),
        "maxdepth": uniform(0.0, 1.0),
    },
}
SEARCH_SPACE_5860 = {
    "name": "5860",
    "positions_hps": {"alpha": 0, "lambda": 1},
    "positions_categorical": {},
    "config_space": {
        "alpha": uniform(0.0, 1.0),
        "lambda": uniform(0.0, 1.0),
    },
}
SEARCH_SPACE_5891 = {
    "name": "5891",
    "positions_hps": {"cost": 0, "gamma": 1, "degree": 3},
    "positions_categorical": {"kernel": [5, 6, 7]},
    "config_space": {
        "cost": uniform(0.0, 1.0),
        "gamma": uniform(0.0, 1.0),
        "degree": uniform(0.0, 1.0),
        # "kernel": choice(["_INVALID", "_linear", "_polynomial"]),
        "kernel": choice([0.0, 1.0, 2.0]),
    },
}
SEARCH_SPACE_5906 = {
    "name": "5906",
    "positions_hps": {
        "eta": 0,
        "max_depth": 1,
        "min_child_weight": 3,
        "subsample": 5,
        "colsample_bytree": 6,
        "colsample_bylevel": 8,
        "lambda": 10,
        "alpha": 11,
        "nrounds": 12,
    },
    "positions_categorical": {"booster": [14, 15]},
    "config_space": {
        "eta": uniform(0.0, 1.0),
        "max_depth": uniform(0.0, 1.0),
        "min_child_weight": uniform(0.0, 1.0),
        "subsample": uniform(0.0, 1.0),
        "colsample_bytree": uniform(0.0, 1.0),
        "colsample_bylevel": uniform(0.0, 1.0),
        "lambda": uniform(0.0, 1.0),
        "alpha": uniform(0.0, 1.0),
        "nrounds": uniform(0.0, 1.0),
        # "booster": choice(["_INVALID", "_gblinear"]),
        "booster": choice([0.0, 1.0]),
    },
}
SEARCH_SPACE_5965 = {
    "name": "5965",
    "positions_hps": {
        "num.trees": 0,
        "sample.fraction": 2,
        "min.node.size": 3,
        "mtry": 4,
    },
    "positions_categorical": {"replace": [6, 7], "respect": [8, 9]},
    "config_space": {
        "num.trees": uniform(0.0, 1.0),
        "mtry": uniform(0.0, 1.0),
        "sample.fraction": uniform(0.0, 1.0),
        "min.node.size": uniform(0.0, 1.0),
        # "replace": choice(["_FALSE", "_INVALID"]),
        "replace": choice([0.0, 1.0]),
        # "respect": choice(["_INVALID", "_TRUE"]),
        "respect": choice([0.0, 1.0]),
    },
}
SEARCH_SPACE_5970 = {
    "name": "5970",
    "positions_hps": {"alpha": 0, "lambda": 1},
    "positions_categorical": {},
    "config_space": {
        "alpha": uniform(0.0, 1.0),
        "lambda": uniform(0.0, 1.0),
    },
}
SEARCH_SPACE_5971 = {
    "name": "5971",
    "positions_hps": {
        "eta": 0,
        "max_depth": 1,
        "min_child_weight": 3,
        "subsample": 5,
        "colsample_bytree": 6,
        "colsample_bylevel": 8,
        "lambda": 10,
        "alpha": 11,
        "nrounds": 12,
    },
    "positions_categorical": {"booster": [14, 15]},
    "config_space": {
        "eta": uniform(0.0, 1.0),
        "max_depth": uniform(0.0, 1.0),
        "min_child_weight": uniform(0.0, 1.0),
        "subsample": uniform(0.0, 1.0),
        "colsample_bytree": uniform(0.0, 1.0),
        "colsample_bylevel": uniform(0.0, 1.0),
        "lambda": uniform(0.0, 1.0),
        "alpha": uniform(0.0, 1.0),
        "nrounds": uniform(0.0, 1.0),
        # "booster": choice(["_INVALID", "_gblinear"]),
        "booster": choice([0.0, 1.0]),
    },
}
SEARCH_SPACE_6766 = {
    "name": "6766",
    "positions_hps": {"alpha": 0, "lambda": 1},
    "positions_categorical": {},
    "config_space": {
        "alpha": uniform(0.0, 1.0),
        "lambda": uniform(0.0, 1.0),
    },
}
SEARCH_SPACE_6767 = {
    "name": "6767",
    "positions_hps": {
        "eta": 0,
        "subsample": 1,
        "lambda": 2,
        "alpha": 3,
        "nthread": 4,
        "nrounds": 6,
        "max_depth": 8,
        "min_child_weight": 10,
        "colsample_bytree": 12,
        "colsample_bylevel": 14,
    },
    "positions_categorical": {"booster": [16, 17]},
    "config_space": {
        "eta": uniform(0.0, 1.0),
        "subsample": uniform(0.0, 1.0),
        "lambda": uniform(0.0, 1.0),
        "alpha": uniform(0.0, 1.0),
        "nthread": uniform(0.0, 1.0),
        "nrounds": uniform(0.0, 1.0),
        "max_depth": uniform(0.0, 1.0),
        "min_child_weight": uniform(0.0, 1.0),
        "colsample_bytree": uniform(0.0, 1.0),
        "colsample_bylevel": uniform(0.0, 1.0),
        # "booster": choice(["_INVALID", "_gblinear"]),
        "booster": choice([0.0, 1.0]),
    },
}
SEARCH_SPACE_6794 = {
    "name": "6794",
    "positions_hps": {
        "num.trees": 0,
        "sample.fraction": 2,
        "min.node.size": 3,
        "mtry": 4,
    },
    "positions_categorical": {"replace": [6, 7], "respect": [8, 9]},
    "config_space": {
        "num.trees": uniform(0.0, 1.0),
        "mtry": uniform(0.0, 1.0),
        "sample.fraction": uniform(0.0, 1.0),
        "min.node.size": uniform(0.0, 1.0),
        # "replace": choice(["_FALSE", "_INVALID"]),
        "replace": choice([0.0, 1.0]),
        # "respect": choice(["_INVALID", "_TRUE"]),
        "respect": choice([0.0, 1.0]),
    },
}
SEARCH_SPACE_7607 = {
    "name": "7607",
    "positions_hps": {
        "num.trees": 0,
        "min.node.size": 2,
        "sample.fraction": 3,
        "mtry": 4,
    },
    "positions_categorical": {"replace": [5, 6], "respect": [7, 8]},
    "config_space": {
        "num.trees": uniform(0.0, 1.0),
        "mtry": uniform(0.0, 1.0),
        "min.node.size": uniform(0.0, 1.0),
        "sample.fraction": uniform(0.0, 1.0),
        # "respect": choice(["_INVALID", "_TRUE"]),
        "respect": choice([0.0, 1.0]),
        # "replace": choice(["_FALSE", "_INVALID"]),
        "replace": choice([0.0, 1.0]),
    },
}
SEARCH_SPACE_7609 = {
    "name": "7609",
    "positions_hps": {
        "num.trees": 0,
        "sample.fraction": 2,
        "min.node.size": 3,
        "mtry": 4,
    },
    "positions_categorical": {"respect": [5, 6], "replace": [7, 8]},
    "config_space": {
        "num.trees": uniform(0.0, 1.0),
        "mtry": uniform(0.0, 1.0),
        "min.node.size": uniform(0.0, 1.0),
        "sample.fraction": uniform(0.0, 1.0),
        # "respect": choice(["_INVALID", "_TRUE"]),
        "respect": choice([0.0, 1.0]),
        # "replace": choice(["_FALSE", "_INVALID"]),
        "replace": choice([0.0, 1.0]),
    },
}
SEARCH_SPACE_5889 = {
    "name": "5889",
    "positions_hps": {"num.trees": 0, "mtry": 1, "sample.fraction": 2, "replace": 3},
    "positions_categorical": {},
    "config_space": {
        "num.trees": uniform(0.0, 1.0),
        "mtry": uniform(0.0, 1.0),
        "sample.fraction": uniform(0.1, 1.0),
        # "replace": choice(["_FALSE", "_INVALID"]),
        "replace": choice([0.0, 1.0]),
    },
}


RESOURCE_ATTR = "hp_epoch"
MAX_RESOURCE_LEVEL = 100


# serialize() and deserialize() had to be overwritten,
# since the HPO-B dataset does not provide the same number of evaluations for each blackbox.
# This is a constraint in the original serialize() function.
def serialize(
    bb_dict: Dict[str, BlackboxTabular], path: str, metadata: Optional[Dict] = None
):
    # check all blackboxes share the objectives
    bb_first = next(iter(bb_dict.values()))
    for bb in bb_dict.values():
        assert bb.objectives_names == bb_first.objectives_names

    path = Path(path)
    path.mkdir(exist_ok=True)

    serialize_configspace(
        path=path,
        configuration_space=bb_first.configuration_space,
    )

    for task, bb in bb_dict.items():
        bb.hyperparameters.to_parquet(
            path / f"{task}-hyperparameters.parquet",
            index=False,
            compression="gzip",
            engine="fastparquet",
        )

        dump_json_with_numpy(
            config_space_to_json_dict(bb_dict[task].fidelity_space),
            filename=path / f"{task}-fidelity_space.json",
        )

        with open(path / f"{task}-objectives_evaluations.npy", "wb") as f:
            np.save(
                f,
                bb_dict[task].objectives_evaluations.astype(np.float32),
                allow_pickle=False,
            )

        with open(path / f"{task}-fidelity_values.npy", "wb") as f:
            np.save(f, bb_dict[task].fidelity_values, allow_pickle=False)

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
    Deserialize blackboxes contained in a path that were saved with ``serialize`` above.
    TODO: the API is currently dissonant with ``serialize``, ``deserialize`` for BlackboxOffline as ``serialize`` is there a member.
    A possible way to unify is to have serialize also be a free function for BlackboxOffline.
    :param path: a path that contains blackboxes that were saved with ``serialize``
    :return: a dictionary from task name to blackbox
    """
    path = Path(path)
    configuration_space, _ = deserialize_configspace(path)
    metadata = deserialize_metadata(path)
    objectives_names = metadata["objectives_names"]
    task_names = metadata["task_names"]

    bb_dict = {}
    for task in task_names:
        hyperparameters = pd.read_parquet(
            Path(path) / f"{task}-hyperparameters.parquet", engine="fastparquet"
        )
        with open(path / f"{task}-fidelity_space.json", "r") as file:
            fidelity_space = config_space_from_json_dict(json.load(file))

        with open(path / f"{task}-fidelity_values.npy", "rb") as f:
            fidelity_values = np.load(f)

        with open(path / f"{task}-objectives_evaluations.npy", "rb") as f:
            objectives_evaluations = np.load(f)

        bb_dict[task] = BlackboxTabular(
            hyperparameters=hyperparameters,
            configuration_space=configuration_space,
            fidelity_space=fidelity_space,
            objectives_evaluations=objectives_evaluations,
            fidelity_values=fidelity_values,
            objectives_names=objectives_names,
        )
    return bb_dict


def generate_hpob(search_space):
    print("generating hpob_" + search_space["name"])
    raw_data_dicts = load_data()
    merged_datasets = merge_multiple_dicts(
        raw_data_dicts[0], raw_data_dicts[1], raw_data_dicts[2]
    )
    df = pd.DataFrame.from_dict(merged_datasets)

    blackbox_name = BLACKBOX_NAME + search_space["name"]
    bb_dict = {}
    indices_not_nan = df[df[search_space["name"]].notna()].index
    for dataset_name in indices_not_nan:
        bb_dict[dataset_name] = convert_dataset(
            search_space, df[search_space["name"]][dataset_name]
        )
    with catchtime("saving to disk"):
        serialize(
            bb_dict=bb_dict,
            path=repository_path / blackbox_name,
            metadata={metric_elapsed_time: METRIC_ELAPSED_TIME},
        )


def convert_dataset(search_space, dataset):
    hp_cols = list(search_space["config_space"].keys())
    n_hps = len(hp_cols)
    n_evals = len(dataset["X"])
    hps = np.zeros((n_evals, n_hps))

    for i, config in enumerate(dataset["X"]):
        # collect all continuous hyperparameters
        final_config = [config[pos] for pos in search_space["positions_hps"].values()]

        # collect categorical hyperparameters and compute encoding
        for positions in search_space.get("positions_categorical", {}).values():
            idx = next(
                (j for j, p in enumerate(positions) if config[p] == 1.0),
                len(positions) - 1,
            )
            final_config.append(float(idx))

        hps[i] = final_config

    hyperparameters = pd.DataFrame(data=hps, columns=hp_cols)

    objective_names = ["metric_accuracy", "metric_elapsed_time"]

    objective_evaluations = np.array(dataset["y"])  # np.array.shape = (N,)
    objective_evaluations = objective_evaluations.reshape(
        objective_evaluations.shape[0], 1, 1, 1
    )

    # Create a metric_elapsed_time array filled with ones as runtime was not provided in the dataset
    elapsed_time_array = np.ones_like(objective_evaluations)

    objective_evaluations = np.concatenate(
        [objective_evaluations, elapsed_time_array], axis=-1
    )
    # fidelity space initialized as constant value, since it is required as an argument
    fidelity_space = {
        RESOURCE_ATTR: randint(lower=MAX_RESOURCE_LEVEL, upper=MAX_RESOURCE_LEVEL)
    }

    return BlackboxTabular(
        hyperparameters=hyperparameters,
        configuration_space=search_space["config_space"],
        objectives_evaluations=objective_evaluations,
        objectives_names=objective_names,
        fidelity_space=fidelity_space,
    )


def load_data():
    hpob_data_file = repository_path / "hpob-data.zip"
    if not hpob_data_file.exists():
        data_src = "https://rewind.tf.uni-freiburg.de/index.php/s/xdrJQPCTNi2zbfL/download/hpob-data.zip"
        print(f"did not find {hpob_data_file}, downloading {data_src}")
        urllib.request.urlretrieve(data_src, hpob_data_file)

        with zipfile.ZipFile(hpob_data_file, "r") as zip_ref:
            zip_ref.extractall(path=repository_path)

    meta_test_file = repository_path / "hpob-data/meta-test-dataset.json"
    meta_train_file = repository_path / "hpob-data/meta-train-dataset.json"
    meta_validation_file = repository_path / "hpob-data/meta-validation-dataset.json"

    with (
        open(meta_test_file, mode="r", encoding="utf-8") as test_file,
        open(meta_train_file, mode="r", encoding="utf-8") as train_file,
        open(meta_validation_file, mode="r", encoding="utf-8") as validation_file,
    ):
        test_data = json.load(test_file)
        train_data = json.load(train_file)
        validation_data = json.load(validation_file)
    return [train_data, validation_data, test_data]


def merge_multiple_dicts(train_dict, validation_dict, test_dict):
    result_dict = train_dict.copy()

    search_spaces = list(train_dict.keys())

    for search_space in search_spaces:
        validation_datasets = list(validation_dict[search_space].keys())
        test_datasets = list(test_dict[search_space].keys())

        for dataset in validation_datasets:
            if dataset in result_dict[search_space].keys():
                result_dict[search_space][dataset]["X"] += validation_dict[
                    search_space
                ][dataset]["X"]
                result_dict[search_space][dataset]["y"] += validation_dict[
                    search_space
                ][dataset]["y"]
            else:
                result_dict[search_space][dataset] = validation_dict[search_space][
                    dataset
                ]

        for dataset in test_datasets:
            if dataset in result_dict[search_space].keys():
                result_dict[search_space][dataset]["X"] += test_dict[search_space][
                    dataset
                ]["X"]
                result_dict[search_space][dataset]["y"] += test_dict[search_space][
                    dataset
                ]["y"]
            else:
                result_dict[search_space][dataset] = test_dict[search_space][dataset]

    return result_dict


class HPOBRecipe(BlackboxRecipe):
    def __init__(self, search_space, name):
        super(HPOBRecipe, self).__init__(
            name=name,
            cite_reference="HPO-B: A Large-Scale Reproducible Benchmark for Black-Box HPO based on OpenML."
            " Sebastian Pineda-Arango and Hadi S. Jomaa and Martin Wistuba and Josif Grabocka, 2021.",
        )
        self.search_space = search_space

    def _generate_on_disk(self):
        generate_hpob(self.search_space)


class HPOBRecipe4796(HPOBRecipe):
    def __init__(self):
        super().__init__(SEARCH_SPACE_4796, "hpob_4796")


class HPOBRecipe5527(HPOBRecipe):
    def __init__(self):
        super().__init__(SEARCH_SPACE_5527, "hpob_5527")


class HPOBRecipe5636(HPOBRecipe):
    def __init__(self):
        super().__init__(SEARCH_SPACE_5636, "hpob_5636")


class HPOBRecipe5859(HPOBRecipe):
    def __init__(self):
        super().__init__(SEARCH_SPACE_5859, "hpob_5859")


class HPOBRecipe5860(HPOBRecipe):
    def __init__(self):
        super().__init__(SEARCH_SPACE_5860, "hpob_5860")


class HPOBRecipe5891(HPOBRecipe):
    def __init__(self):
        super().__init__(SEARCH_SPACE_5891, "hpob_5891")


class HPOBRecipe5906(HPOBRecipe):
    def __init__(self):
        super().__init__(SEARCH_SPACE_5906, "hpob_5906")


class HPOBRecipe5965(HPOBRecipe):
    def __init__(self):
        super().__init__(SEARCH_SPACE_5965, "hpob_5965")


class HPOBRecipe5970(HPOBRecipe):
    def __init__(self):
        super().__init__(SEARCH_SPACE_5970, "hpob_5970")


class HPOBRecipe5971(HPOBRecipe):
    def __init__(self):
        super().__init__(SEARCH_SPACE_5971, "hpob_5971")


class HPOBRecipe6766(HPOBRecipe):
    def __init__(self):
        super().__init__(SEARCH_SPACE_6766, "hpob_6766")


class HPOBRecipe6767(HPOBRecipe):
    def __init__(self):
        super().__init__(SEARCH_SPACE_6767, "hpob_6767")


class HPOBRecipe6794(HPOBRecipe):
    def __init__(self):
        super().__init__(SEARCH_SPACE_6794, "hpob_6794")


class HPOBRecipe7607(HPOBRecipe):
    def __init__(self):
        super().__init__(SEARCH_SPACE_7607, "hpob_7607")


class HPOBRecipe7609(HPOBRecipe):
    def __init__(self):
        super().__init__(SEARCH_SPACE_7609, "hpob_7609")


class HPOBRecipe5889(HPOBRecipe):
    def __init__(self):
        super().__init__(SEARCH_SPACE_5889, "hpob_5889")


if __name__ == "__main__":
    recipes = [
        HPOBRecipe4796,
        HPOBRecipe5527,
        HPOBRecipe5636,
        HPOBRecipe5859,
        HPOBRecipe5860,
        HPOBRecipe5891,
        HPOBRecipe5906,
        HPOBRecipe5965,
        HPOBRecipe5970,
        HPOBRecipe5971,
        HPOBRecipe6766,
        HPOBRecipe6767,
        HPOBRecipe6794,
        HPOBRecipe7607,
        HPOBRecipe7609,
        HPOBRecipe5889,
    ]

    for recipe in recipes:
        instance = recipe()
        instance.generate(upload_on_hub=True)
