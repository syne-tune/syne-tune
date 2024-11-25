import zipfile
import urllib.request
import json
import pandas as pd
import numpy as np

from syne_tune.blackbox_repository.blackbox_tabular import serialize, BlackboxTabular
from syne_tune.blackbox_repository.conversion_scripts.utils import (
    repository_path,
)
from syne_tune.blackbox_repository.conversion_scripts.blackbox_recipe import (
    BlackboxRecipe,
)
from syne_tune.config_space import uniform, loguniform, randint
from syne_tune.util import catchtime

BLACKBOX_NAME = "hpob_"

# configuration_space values taken from "https://raw.githubusercontent.com/machinelearningnuremberg/HPO-B/refs/heads/main/hpob-data/meta-dataset-descriptors.json"
SEARCH_SPACE_4796 = {
    "name": "4796",
    "config_space": {
        "minsplit": loguniform(2.0, 128.0),
        "minbucket": loguniform(1.0, 64.0),
        "cp": loguniform(0.0002, 0.1),
    }
}

SEARCH_SPACE_5527 = {
    "name": "5527",
    "config_space": {
        "cost": loguniform(0.001, 1024.0),
        "gamma": loguniform(0.0001, 1024.0),
        "gamma.na": uniform(0.0, 1.0),
        "degree": uniform(0.0, 5.0),
        "degree.na": uniform(0.0, 1.0),
        "kernel.ohe.na": 308190.0,
        "kernel.ohe.linear": 316563.0,
        "kernel.ohe.polynomial": 311266.0,
    },
}
SEARCH_SPACE_5636 = {
    "name": "5636",
    "config_space": {
        "minsplit": uniform(0.0, 60.0),
        "minsplit.na": uniform(0.0, 1.0),
        "minbucket": uniform(1.0, 60.0),
        "cp": uniform(-9.2, 9.906167518025702e-5),
        "maxdepth": uniform(0.0, 29.0),
        "maxdepth.na": uniform(0.0, 1.0),
    },
}
SEARCH_SPACE_5859 = {
    "name": "5859",
    "config_space": {
        "minsplit": uniform(0.0, 60.0),
        "minsplit.na": uniform(0.0, 1.0),
        "minbucket": uniform(1.0, 60.0),
        "cp": loguniform(0.0001, 1.0),
        "maxdepth": uniform(0.0, 29.0),
        "maxdepth.na": uniform(0.0, 1.0),
    },
}
SEARCH_SPACE_5860 = {
    "name": "5860",
    "config_space": {
        "alpha": loguniform(0.000578, 1.0),
        "lambda": loguniform(0.001, 1019.35),
    },
}
SEARCH_SPACE_5891 = {
    "name": "5891",
    "config_space": {
        "cost": uniform(0.001, 1024.0),
        "gamma": uniform(0.0, 1024.0),
        "gamma.na": uniform(0.0, 1.0),
        "degree": uniform(0.0, 5.0),
        "degree.na": uniform(0.0, 1.0),
        "kernel.ohe.na": 24627,
        "kernel.ohe.linear": 24952,
        "kernel.ohe.polynomial": 24813,
    },
}
SEARCH_SPACE_5906 = {
    "name": "5906",
    "config_space": {
        "eta": loguniform(0.001, 1.0),
        "max_depth": uniform(0.0, 15.0),
        "max_depth.na": uniform(0.0, 1.0),
        "min_child_weight": loguniform(0.0001, 128.0),
        "min_child_weight.na": uniform(0.0, 1.0),
        "subsample": uniform(0.1, 1.0),
        "colsample_bytree": loguniform(0.0001, 1.0),
        "colsample_bytree.na": uniform(0.0, 1.0),
        "colsample_bylevel": uniform(0.0, 1.0),
        "colsample_bylevel.na": uniform(0.0, 1.0),
        "lambda": loguniform(0.001, 1020.8),
        "alpha": loguniform(0.001, 1024.0),
        "nrounds": loguniform(0.0001, 5000.0),
        "nrounds.na": uniform(0.0, 1.0),
        "booster.ohe.na": 1449,
        "booster.ohe.gblinear": 1937,
    },
}
SEARCH_SPACE_5965 = {
    "name": "5965",
    "config_space": {
        "num.trees": loguniform(0.0001, 2000.0),
        "num.trees.na": uniform(0.0, 1.0),
        "mtry": loguniform(1.0, 1776.0),
        "sample.fraction": uniform(0.1, 1.0),
        "min.node.size": loguniform(0.0001, 45310.0),
        "min.node.size.na": uniform(0.0, 1.0),
        "replace.ohe.FALSE": 295436,
        "replace.ohe.na": 296498,
        "respect.unordered.factors.ohe.INVALID": 294378,
        "respect.unordered.factors.ohe.TRUE": 297556,
    },
}
SEARCH_SPACE_5970 = {
    "name": "5970",
    "config_space": {
        "alpha": loguniform(1.0, 2.72),
        # TODO check these values again, were they in logspace already?
        "lambda": loguniform(0.001, 1024.0),
    },
}
SEARCH_SPACE_5971 = {
    "name": "5971",
    "config_space": {
        "eta": loguniform(0.001, 1.0),
        "max_depth": uniform(0.0, 15.0),
        "max_depth.na": uniform(0.0, 1.0),
        "min_child_weight": loguniform(0.0001, 128.0),
        "min_child_weight.na": uniform(0.0, 1.0),
        "subsample": uniform(0.1, 1.0),
        "colsample_bytree": uniform(0.0, 1.0),
        "colsample_bytree.na": uniform(0.0, 1.0),
        "colsample_bylevel": uniform(0.0, 1.0),
        "colsample_bylevel.na": uniform(0.0, 1.0),
        "lambda": loguniform(0.001, 1024.0),
        "alpha": loguniform(0.001, 1024.0),
        "nrounds": uniform(0, 5000),
        "nrounds.na": uniform(0.0, 1.0),
        "booster.ohe.na": 34130,
        "booster.ohe.gblinear": 45096,
    },
}
SEARCH_SPACE_6766 = {
    "name": "6766",
    "config_space": {
        "alpha": loguniform(1.0, 2.72),
        # TODO again here, is this already in logspace?
        "lambda": loguniform(0.001, 1024.0),
    },
}
SEARCH_SPACE_6767 = {
    "name": "6767",
    "config_space": {
        "eta": uniform(0.001, 1.0),
        "subsample": uniform(0.1, 1.0),
        "lambda": uniform(0.001, 1024.0),
        "alpha": uniform(0.001, 1024.0),
        "nthread": uniform(0.0, 1.0),
        "nthread.na": uniform(0.0, 1.0),
        "nrounds": uniform(0.0, 5000.0),
        "nrounds.na": uniform(0.0, 1.0),
        "max_depth": uniform(0.0, 15.0),
        "max_depth.na": uniform(0.0, 1.0),
        "min_child_weight": uniform(0.0, 128.0),
        "min_child_weight.na": uniform(0.0, 1.0),
        "colsample_bytree": uniform(0.0, 1.0),
        "colsample_bytree.na": uniform(0.0, 1.0),
        "colsample_bylevel": uniform(0.0, 1.0),
        "colsample_bylevel.na": uniform(0.0, 1.0),
        "booster.ohe.na": 430736,
        "booster.ohe.gblinear": 620644,
    },
}
SEARCH_SPACE_6794 = {
    "name": "6794",
    "config_space": {
        "num.trees": uniform(0.0, 2000.0),
        "num.trees.na": uniform(0.0, 1.0),
        "mtry": uniform(1.0, 1558.0),
        "sample.fraction": uniform(0.1, 1.0),
        "min.node.size": uniform(-9.21, 10.72),
        "min.node.size.na": uniform(0.0, 3195.0),
        "replace.ohe.FALSE": 623218,
        "replace.ohe.na": 625345,
        "respect.unordered.factors.ohe.INVALID": 625034,
        "respect.unordered.factors.ohe.TRUE": 623529,
    },
}
SEARCH_SPACE_7607 = {
    "name": "7607",
    "config_space": {
        "num.trees": uniform(0.0, 2000.0),
        "num.trees.na": uniform(0.0, 1.0),
        "mtry": uniform(1.0, 1775.0),
        "min.node.size": loguniform(1.0, 44808.0),
        "sample.fraction": uniform(0.1, 1.0),
        "replace.ohe.FALSE": 14478,
        "replace.ohe.na": 14434,
        "respect.unordered.factors.ohe.INVALID": 14392,
        "respect.unordered.factors.ohe.TRUE": 14520,
    },
}
SEARCH_SPACE_7609 = {
    "name": "7609",
    "config_space": {
        "num.trees": uniform(0.0, 2000.0),
        "num.trees.na": uniform(0.0, 1.0),
        "mtry": uniform(1.0, 1776.0),
        "min.node.size": loguniform(1.0, 45280.0),
        "sample.fraction": uniform(0.1, 1.0),
        "replace.ohe.FALSE": 32943,
        "replace.ohe.na": 33003,
        "respect.unordered.factors.ohe.INVALID": 33007,
        "respect.unordered.factors.ohe.TRUE": 32939,
    },
}
SEARCH_SPACE_5889 = {
    "name": "5889",
    "config_space": {
        "num.trees": loguniform(0.0001, 2000.0),
        "num.trees.na": uniform(0.0, 1.0),
        "mtry": loguniform(1.0, 1324.0),
        "sample.fraction": uniform(0.1, 1.0),
        "replace.ohe.FALSE": 1215,
        "replace.ohe.na": 1226,
    },
}
RESOURCE_ATTR = "hp_epoch"
MAX_RESOURCE_LEVEL = 100


def generate_hpob(search_space):
    print("generating hpob")
    raw_data_dicts = load_data()
    merged_datasets = merge_multiple_dicts(raw_data_dicts[0], raw_data_dicts[1], raw_data_dicts[2])
    df = pd.DataFrame.from_dict(merged_datasets)

    blackbox_name = BLACKBOX_NAME + search_space["name"]
    bb_dict = {}
    indices_not_nan = df[df[search_space["name"]].notna()].index
    for dataset_name in indices_not_nan:
        bb_dict[dataset_name] = convert_dataset(
            search_space, dataset_name, df[search_space["name"]][dataset_name]
        )
    with catchtime("saving to disk"):
        serialize(
            bb_dict=bb_dict,
            path=repository_path / blackbox_name,
        )


def convert_dataset(search_space, dataset_name, dataset):
    hp_cols = list(search_space["config_space"].keys())
    n_hps = len(hp_cols)
    n_evals = len(dataset['X'])
    hps = np.zeros((n_evals, n_hps))

    for i, config in enumerate(dataset["X"]):
        hps[i] = config  # Directly assign each config to the corresponding row in hps

    hyperparameters = pd.DataFrame(data=hps, columns=hp_cols)

    objective_names = ["metric_accuracy"]

    objective_evaluations = np.array(dataset['y'])  # np.array.shape = (N,)
    objective_evaluations = objective_evaluations.reshape(objective_evaluations.shape[0], 1, 1, 1)

    fidelity_space = {RESOURCE_ATTR: randint(lower=1, upper=MAX_RESOURCE_LEVEL)}

    return BlackboxTabular(
        hyperparameters=hyperparameters,
        configuration_space=search_space["config_space"],
        objectives_evaluations=objective_evaluations,
        objectives_names=objective_names,
        fidelity_space=fidelity_space
    )


def load_data():
    hpob_data_file = repository_path / "hpob-data.zip"
    if not hpob_data_file.exists():
        data_src = "https://rewind.tf.uni-freiburg.de/index.php/s/xdrJQPCTNi2zbfL/download/hpob-data.zip"
        print(f"did not find {hpob_data_file}, downloading {data_src}")
        urllib.request.urlretrieve(data_src, hpob_data_file)

        with zipfile.ZipFile(hpob_data_file, "r") as zip_ref:
            zip_ref.extractall()

    meta_test_file = "hpob-data/meta-test-dataset.json"
    meta_train_file = "hpob-data/meta-train-dataset.json"
    meta_validation_file = "hpob-data/meta-validation-dataset.json"

    # search spaces: 4796, 5527, 5636, 5859, 5860, 5891, 5906, 5965, 5970, 5971, 6766, 6767, 6794, 7607, 7609, 5889
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

    for dataset in result_dict['4796']:
        print(len(result_dict['4796'][dataset]['X']))
    search_spaces = list(train_dict.keys())

    for search_space in search_spaces:
        validation_datasets = list(validation_dict[search_space].keys())
        test_datasets = list(test_dict[search_space].keys())

        for dataset in validation_datasets:
            if dataset in result_dict[search_space].keys():
                result_dict[search_space][dataset]['X'] += validation_dict[search_space][dataset]['X']
                result_dict[search_space][dataset]['y'] += validation_dict[search_space][dataset]['y']
            else:
                result_dict[search_space][dataset] = validation_dict[search_space][dataset]

        for dataset in test_datasets:
            if dataset in result_dict[search_space].keys():
                result_dict[search_space][dataset]['X'] += test_dict[search_space][dataset]['X']
                result_dict[search_space][dataset]['y'] += test_dict[search_space][dataset]['y']
            else:
                result_dict[search_space][dataset] = test_dict[search_space][dataset]

    summe_eval = 0
    datasets = 0
    for i in result_dict['4796']:
        datasets += 1
        summe_eval += len(result_dict['4796'][i]['X'])
    print("eval ", summe_eval)
    print("datasets", datasets)
    return result_dict


class HPOBRecipe(BlackboxRecipe):
    def __init__(self):
        super(HPOBRecipe, self).__init__(
            name="hpob_4796",
            cite_reference="HPO-B: A Large-Scale Reproducible Benchmark for Black-Box HPO based on OpenML."
                           "Sebastian Pineda-Arango and Hadi S. Jomaa and Martin Wistuba and Josif Grabocka, 2021.",
        )

    def _generate_on_disk(self):
        generate_hpob(SEARCH_SPACE_4796)


# class HPORecipe4796(HPOBRecipe):
#    super(name='HPOB_4796')


if __name__ == "__main__":
    HPOBRecipe().generate()
