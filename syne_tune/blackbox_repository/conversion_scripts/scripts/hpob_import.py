import zipfile
import urllib.request
import json
import pandas as pd
import numpy as np

from pathlib import Path

from syne_tune.blackbox_repository.blackbox_tabular import serialize, BlackboxTabular
from syne_tune.blackbox_repository.conversion_scripts.utils import (
    repository_path,
)
from syne_tune.blackbox_repository.conversion_scripts.blackbox_recipe import (
    BlackboxRecipe,
)
from syne_tune.config_space import uniform, loguniform
from syne_tune.util import catchtime

BLACKBOX_NAME = "hpob_"

# configuration_space values taken from "https://raw.githubusercontent.com/machinelearningnuremberg/HPO-B/refs/heads/main/hpob-data/meta-dataset-descriptors.json"
SEARCH_SPACES = [
    {
        'name': '4796',
        'config_space': {
            "minsplit": loguniform(2.0001, 128.0001),
            "minbucket": loguniform(1.0001, 64.00010000000003),
            "cp": loguniform(0.00020009788511334296, 0.10007952104999131)}
    },
    {
        'name': '5527',
        'config_space': {
            "cost": loguniform(0.0010765753825886914, 1023.98985728813),
            "gamma": loguniform(0.00010000000000000009, 1023.8675343735601),
            "gamma.na": uniform(0.0, 1.0),
            "degree": uniform(0.0, 5.0),
            "degree.na": uniform(0.0, 1.0),
            "kernel.ohe.na": 308190,
            "kernel.ohe.linear": 316563,
            "kernel.ohe.polynomial": 311266
        }},
    {'name': '5636',
     'config_space': {
         "minsplit": uniform(0.0, 60.0),
         "minsplit.na": uniform(0.0, 1.0),
         "minbucket": uniform(1.0, 60.0),
         "cp": uniform(-9.2054906470863, 9.906167518025702e-5),
         "maxdepth": uniform(0.0, 29.0),
         "maxdepth.na": uniform(0.0, 1.0)
     }},
    {'name': '5859',
     'config_space': {
         "minsplit": uniform(0.0, 60.0),
         "minsplit.na": uniform(0.0, 1.0),
         "minbucket": uniform(1.0, 60.0),
         "cp": loguniform(0.00010078883022069933, 1.000092678873241),
         "maxdepth": uniform(0.0, 29.0),
         "maxdepth.na": uniform(0.0, 1.0)
     }},
    {'name': '5860',
     'config_space': {
         "alpha": loguniform(0.0005784537013620139, 1.000052937941998),
         "lambda": loguniform(0.0010772863790518194, 1019.34589896755)
     }},

    {'name': '5891',
     'config_space': {
         "cost": uniform(0.000976654787468175, 1023.8986416547),
         "gamma": uniform(0.0, 1023.1136362028),
         "gamma.na": uniform(0.0, 1.0),
         "degree": uniform(0.0, 5.0),
         "degree.na": uniform(0.0, 1.0),
         "kernel.ohe.na": 24627,
         "kernel.ohe.linear": 24952,
         "kernel.ohe.polynomial": 24813
     }},
    {'name': '5906',
     'config_space': {
         "eta": loguniform(0.0010768343186195047, 0.999989716568596),
         "max_depth": uniform(0.0, 15.0),
         "max_depth.na": uniform(0.0, 1.0),
         "min_child_weight": loguniform(0.00010000000000000009, 127.637710948085),
         "min_child_weight.na": uniform(0.0, 1.0),
         "subsample": uniform(0.100336084561422, 0.99988544494845),
         "colsample_bytree": loguniform(0.00010000000000000009, 0.9998821955475959),
         "colsample_bytree.na": uniform(0.0, 1.0),
         "colsample_bylevel": uniform(0.0, 0.99936268129386),
         "colsample_bylevel.na": uniform(0.0, 1.0),
         "lambda": loguniform(0.0010785244084924811, 1020.7715708697797),
         "alpha": loguniform(0.0010770880186598605, 1023.0100353264299),
         "nrounds": loguniform(0.00010000000000000009, 5000.000100000004),
         "nrounds.na": uniform(0.0, 1.0),
         "booster.ohe.na": 1449,
         "booster.ohe.gblinear": 1937
     }},
    {'name': '5965',
     'config_space': {
         "num.trees": loguniform(0.00010000000000000009, 2000.0000999999997),
         "num.trees.na": uniform(0.0, 1.0),
         "mtry": loguniform(1.0001, 1776.0000999999997),
         "sample.fraction": uniform(0.100001647463068, 0.999996356386691),
         "min.node.size": loguniform(0.00010000000000000009, 45310.00010000002),
         "min.node.size.na": uniform(0.0, 1.0),
         "replace.ohe.FALSE": 295436,
         "replace.ohe.na": 296498,
         "respect.unordered.factors.ohe.INVALID": 294378,
         "respect.unordered.factors.ohe.TRUE": 297556
     }},
    {'name': '5970',
     'config_space': {
         "alpha": loguniform(1.0000014251573854, 2.7182494957244043),
         # TODO check these values again, were they in logspace already?
         "lambda": loguniform(0.000976593163803205, 1023.93132059391)
     }},
    {'name': '5971',
     'config_space': {
         "eta": loguniform(0.0010766616470413745, 1.0000192154854068),
         "max_depth": uniform(0.0, 15.0),
         "max_depth.na": uniform(0.0, 1.0),
         "min_child_weight": loguniform(0.00010000000000000009, 127.96443821912703),
         "min_child_weight.na": uniform(0.0, 1.0),
         "subsample": uniform(0.100020958529785, 0.999991231691092),
         "colsample_bytree": uniform(0.0, 0.999970588600263),
         "colsample_bytree.na": uniform(0.0, 1.0),
         "colsample_bylevel": uniform(0.0, 0.999982544919476),
         "colsample_bylevel.na": uniform(0.0, 1.0),
         "lambda": loguniform(0.0010766378152753842, 1023.8406670644002),
         "alpha": loguniform(0.0010767250341730224, 1023.98189190699),
         "nrounds": uniform(0, 5000),
         "nrounds.na": uniform(0.0, 1.0),
         "booster.ohe.na": 34130,
         "booster.ohe.gblinear": 45096
     }},
    {'name': '6766',
     'config_space': {
         "alpha": loguniform(1.000000123167418, 2.7182804550678936),
         # TODO again here, is this already in logspace?
         "lambda": loguniform(0.00097656444798024, 1023.99289718568)
     }},
    {'name': '6767',
     'config_space': {
         "eta": uniform(0.000976579456699394, 0.999989898907273),
         "subsample": uniform(0.100003587454557, 0.999998953519389),
         "lambda": uniform(0.000976577472439872, 1023.97805712306),
         "alpha": uniform(0.0009765784401402, 1023.97382658781),
         "nthread": uniform(0.0, 1.0),
         "nthread.na": uniform(0.0, 1.0),
         "nrounds": uniform(0.0, 5000.0),
         "nrounds.na": uniform(0.0, 1.0),
         "max_depth": uniform(0.0, 15.0),
         "max_depth.na": uniform(0.0, 1.0),
         "min_child_weight": uniform(0.0, 127.996673624102),
         "min_child_weight.na": uniform(0.0, 1.0),
         "colsample_bytree": uniform(0.0, 0.999999715248123),
         "colsample_bytree.na": uniform(0.0, 1.0),
         "colsample_bylevel": uniform(0.0, 0.999995714752004),
         "colsample_bylevel.na": uniform(0.0, 1.0),
         "booster.ohe.na": 430736,
         "booster.ohe.gblinear": 620644
     }},
    {'name': '6794',
     'config_space': {
         "num.trees": uniform(0.0, 2000.0),
         "num.trees.na": uniform(0.0, 1.0),
         "mtry": uniform(1.0, 1558.0),
         "sample.fraction": uniform(0.100001274677925, 0.999999188841321),
         "min.node.size": uniform(-9.210340371976182, 10.721283039868203),
         "min.node.size.na": uniform(0.0, 3195.0),
         "replace.ohe.FALSE": 623218,
         "replace.ohe.na": 625345,
         "respect.unordered.factors.ohe.INVALID": 625034,
         "respect.unordered.factors.ohe.TRUE": 623529
     }},
    {'name': '7607',
     'config_space': {
         "num.trees": uniform(0.0, 2000.0),
         "num.trees.na": uniform(0.0, 1.0),
         "mtry": uniform(1.0, 1775.0),
         "min.node.size": loguniform(1.0001, 44808.00010000003),
         "sample.fraction": uniform(0.10000844351016, 0.999976679659449),
         "replace.ohe.FALSE": 14478,
         "replace.ohe.na": 14434,
         "respect.unordered.factors.ohe.INVALID": 14392,
         "respect.unordered.factors.ohe.TRUE": 14520
     }},
    {'name': '7609',
     'config_space': {
         "num.trees": uniform(0.0, 2000.0),
         "num.trees.na": uniform(0.0, 1.0),
         "mtry": uniform(1.0, 1776.0),
         "min.node.size": loguniform(1.0001, 45280.00009999999),
         "sample.fraction": uniform(0.10002462414559, 0.999995367531665),
         "replace.ohe.FALSE": 32943,
         "replace.ohe.na": 33003,
         "respect.unordered.factors.ohe.INVALID": 33007,
         "respect.unordered.factors.ohe.TRUE": 32939
     }},
    {'name': '5889',
     'config_space': {
         "num.trees": loguniform(0.00010000000000000009, 2000.0000999999997),
         "num.trees.na": uniform(0.0, 1.0),
         "mtry": loguniform(1.0001, 1324.0001),
         "sample.fraction": uniform(0.10086305460427, 0.999975134665146),
         "replace.ohe.FALSE": 1215,
         "replace.ohe.na": 1226,
     }}
]


def generate_hpob(search_space, df):
    print("generating hpob")
    blackbox_name = BLACKBOX_NAME + search_space['name']
    bb_dict = {}

    for dataset in df[search_space['name']]:
        bb_dict[search_space['name']][dataset] = convert_dataset(search_space, df, dataset)

    with catchtime("saving to disk"):
        serialize(
            bb_dict=bb_dict,
            path=repository_path / blackbox_name,
        )


def convert_dataset(search_space, df, dataset):
    hp_cols = list(search_space['config_space'].keys())
    n_hps = len(hp_cols)
    n_evals = len(df[search_space['name']][dataset]['X'])
    hps = np.ndarray(shape=(n_evals, n_hps))
    for config in df[search_space['name']][dataset]['X']:
        np.append(hps, config)

    hyperparameters = pd.DataFrame(data=hps, columns=hp_cols)

    objective_names = ["metric_accuracy"]
    n_objectives = len(objective_names)
    n_seeds = 1
    n_fidelities = 1

    objective_evaluations = np.empty(
        (n_evals, n_seeds, n_fidelities, n_objectives)
    ).astype("float32")

    evaluations = np.ndarray(shape=(n_evals, n_hps))
    for evaluation in df[search_space['name']][dataset]['y']:
        np.append(evaluations, evaluation)

    for i, eval_value in enumerate(evaluations):
        objective_evaluations[i, n_seeds, n_fidelities, n_objectives] = eval_value

    return BlackboxTabular(
        hyperparameters=hyperparameters,
        configuration_space=search_space['config_space'],
        objectives_evaluations=objective_evaluations,
        objectives_names=objective_names,
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

    # 4796, 5527, 5636, 5859, 5860, 5891, 5906, 5965, 5970, 5971, 6766, 6767, 6794, 7607, 7609, 5889
    with (open(meta_test_file, mode="r", encoding="utf-8") as test_file,
          open(meta_train_file, mode="r", encoding="utf-8") as train_file,
          open(meta_validation_file, mode="r", encoding="utf-8") as validation_file):
        test_data = json.load(test_file)
        train_data = json.load(train_file)
        validation_data = json.load(validation_file)
    return [train_data, validation_data, test_data]


def merge_multiple_dicts(train_dict, *dicts):
    # Initialize result_dict with a copy of the first dictionary to start with
    result_dict = train_dict.copy()

    # Collect all top-level keys across all dictionaries
    all_keys = set(key for d in dicts for key in d.keys())

    # Process each top-level key
    for key in all_keys:
        # Initialize key in result_dict if not present
        if key not in result_dict:
            result_dict[key] = {}

        # Loop over each input dictionary
        for d in dicts:
            if key not in d:
                continue  # Skip if the current key is not in the dictionary

            for sub_key, sub_value in d[key].items():
                # If sub_key already exists, extend the 'X' and 'y' lists
                if sub_key in result_dict[key]:
                    result_dict[key][sub_key]['X'].extend(sub_value['X'])
                    result_dict[key][sub_key]['y'].extend(sub_value['y'])
                else:
                    # Otherwise, directly assign the sub_key data
                    result_dict[key][sub_key] = sub_value

    return result_dict


class HPOBRecipe(BlackboxRecipe):
    def __init__(self):
        super(HPOBRecipe, self).__init__(
            name=BLACKBOX_NAME,
            cite_reference="HPO-B: A Large-Scale Reproducible Benchmark for Black-Box HPO based on OpenML."
                           "Sebastian Pineda-Arango and Hadi S. Jomaa and Martin Wistuba and Josif Grabocka, 2021.",
        )

    def _generate_on_disk(self):
        raw_data_dicts = load_data()
        merged_datasets = merge_multiple_dicts(raw_data_dicts[0], raw_data_dicts[1:])
        df = pd.DataFrame.from_dict(merged_datasets)
        for search_space in SEARCH_SPACES:
            generate_hpob(search_space, df)


if __name__ == "__main__":
    HPOBRecipe().generate()
