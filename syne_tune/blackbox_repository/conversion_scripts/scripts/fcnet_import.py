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
"""
Convert tabular data from
 Tabular Benchmarks for Joint Architecture and Hyperparameter Optimization
 Aaron Klein Frank Hutter
 https://arxiv.org/pdf/1905.04970.pdf.
"""
import urllib
import tarfile

from pathlib import Path
import pandas as pd
import numpy as np
import ast

try:
    import h5py
except ImportError:
    print("Cannot import h5py. Use 'pip install h5py'")

from syne_tune.blackbox_repository.blackbox_tabular import serialize, BlackboxTabular
from syne_tune.blackbox_repository.conversion_scripts.blackbox_recipe import (
    BlackboxRecipe,
)
from syne_tune.blackbox_repository.conversion_scripts.scripts import (
    metric_elapsed_time,
    default_metric,
    resource_attr,
)
from syne_tune.blackbox_repository.conversion_scripts.utils import repository_path

from syne_tune.util import catchtime
from syne_tune.config_space import choice, logfinrange, finrange, randint

BLACKBOX_NAME = "fcnet"

METRIC_VALID_LOSS = "metric_valid_loss"

METRIC_ELAPSED_TIME = "metric_elapsed_time"

RESOURCE_ATTR = "hp_epoch"

MAX_RESOURCE_LEVEL = 100

NUM_UNITS_1 = "hp_n_units_1"

NUM_UNITS_2 = "hp_n_units_2"

CONFIGURATION_SPACE = {
    "hp_activation_fn_1": choice(["tanh", "relu"]),
    "hp_activation_fn_2": choice(["tanh", "relu"]),
    "hp_batch_size": logfinrange(8, 64, 4, cast_int=True),
    "hp_dropout_1": finrange(0.0, 0.6, 3),
    "hp_dropout_2": finrange(0.0, 0.6, 3),
    "hp_init_lr": choice([0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]),
    "hp_lr_schedule": choice(["cosine", "const"]),
    NUM_UNITS_1: logfinrange(16, 512, 6, cast_int=True),
    NUM_UNITS_2: logfinrange(16, 512, 6, cast_int=True),
}


def convert_dataset(dataset_path: Path, max_rows: int = None):
    data = h5py.File(dataset_path, "r")
    keys = data.keys()
    if max_rows is not None:
        keys = list(keys)[:max_rows]

    hyperparameters = pd.DataFrame(ast.literal_eval(key) for key in keys)
    hyperparameters.rename(
        columns={col: "hp_" + col for col in hyperparameters.columns}, inplace=True
    )

    objective_names = [
        "valid_loss",
        "train_loss",
        "final_test_error",
        "n_params",
        "elapsed_time",
    ]

    # todo for now only full metrics
    fidelity_values = np.arange(1, MAX_RESOURCE_LEVEL + 1)
    n_fidelities = len(fidelity_values)
    n_objectives = len(objective_names)
    n_seeds = 4
    n_hps = len(keys)

    objective_evaluations = np.empty(
        (n_hps, n_seeds, n_fidelities, n_objectives)
    ).astype("float32")

    def save_objective_values_helper(name, values):
        assert values.shape == (n_hps, n_seeds, n_fidelities)

        name_pos = objective_names.index(name)
        objective_evaluations[..., name_pos] = values

    # (n_hps, n_seeds,)
    final_test_error = np.stack(
        [data[key]["final_test_error"][:].astype("float32") for key in keys]
    )

    # (n_hps, n_seeds, n_fidelities)
    final_test_error = np.repeat(
        np.expand_dims(final_test_error, axis=-1), n_fidelities, axis=-1
    )
    save_objective_values_helper("final_test_error", final_test_error)

    # (n_hps, n_seeds,)
    n_params = np.stack([data[key]["n_params"][:].astype("float32") for key in keys])

    # (n_hps, n_seeds, n_fidelities)
    n_params = np.repeat(np.expand_dims(n_params, axis=-1), n_fidelities, axis=-1)
    save_objective_values_helper("n_params", n_params)

    # (n_hps, n_seeds,)
    runtime = np.stack([data[key]["runtime"][:].astype("float32") for key in keys])

    # linear interpolation to go from total training time to training time per epoch as in fcnet code
    # (n_hps, n_seeds, n_epochs)
    # todo utilize expand dim instead of reshape
    epochs = np.repeat(fidelity_values.reshape(1, -1), n_hps * n_seeds, axis=0).reshape(
        n_hps, n_seeds, -1
    )
    elapsed_time = (epochs / MAX_RESOURCE_LEVEL) * runtime.reshape((n_hps, n_seeds, 1))

    save_objective_values_helper("elapsed_time", elapsed_time)

    # metrics that are fully observed, only use train/valid loss as mse are the same numbers
    # for m in ['train_loss', 'train_mse', 'valid_loss', 'valid_mse']:
    for m in ["train_loss", "valid_loss"]:
        save_objective_values_helper(
            m, np.stack([data[key][m][:].astype("float32") for key in keys])
        )

    fidelity_space = {RESOURCE_ATTR: randint(lower=1, upper=MAX_RESOURCE_LEVEL)}

    objective_names = [f"metric_{m}" for m in objective_names]
    # Sanity checks:
    assert objective_names[0] == METRIC_VALID_LOSS
    assert objective_names[4] == METRIC_ELAPSED_TIME
    return BlackboxTabular(
        hyperparameters=hyperparameters,
        configuration_space=CONFIGURATION_SPACE,
        fidelity_space=fidelity_space,
        objectives_evaluations=objective_evaluations,
        fidelity_values=fidelity_values,
        objectives_names=objective_names,
    )


def generate_fcnet():
    blackbox_name = BLACKBOX_NAME
    fcnet_file = repository_path / "fcnet_tabular_benchmarks.tar.gz"
    if not fcnet_file.exists():
        src = "http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz"
        print(f"did not find {fcnet_file}, downloading {src}")
        urllib.request.urlretrieve(src, fcnet_file)

    with tarfile.open(fcnet_file) as f:
        f.extractall(path=repository_path)

    with catchtime("converting"):
        bb_dict = {}
        for dataset in [
            "protein_structure",
            "naval_propulsion",
            "parkinsons_telemonitoring",
            "slice_localization",
        ]:
            print(f"converting {dataset}")
            dataset_path = (
                repository_path
                / "fcnet_tabular_benchmarks"
                / f"fcnet_{dataset}_data.hdf5"
            )
            bb_dict[dataset] = convert_dataset(dataset_path=dataset_path)

    with catchtime("saving to disk"):
        serialize(
            bb_dict=bb_dict,
            path=repository_path / blackbox_name,
            metadata={
                metric_elapsed_time: METRIC_ELAPSED_TIME,
                default_metric: METRIC_VALID_LOSS,
                resource_attr: RESOURCE_ATTR,
            },
        )


def plot_learning_curves():
    import matplotlib.pyplot as plt
    from syne_tune.blackbox_repository.repository import load_blackbox

    # plot one learning-curve for sanity-check
    bb_dict = load_blackbox(BLACKBOX_NAME)

    b = bb_dict["naval_propulsion"]
    configuration = {k: v.sample() for k, v in b.configuration_space.items()}
    print(configuration)
    errors = []
    for i in range(1, MAX_RESOURCE_LEVEL + 1):
        res = b.objective_function(configuration=configuration, fidelity={"epochs": i})
        errors.append(res[METRIC_VALID_LOSS])
    plt.plot(errors)


class FCNETRecipe(BlackboxRecipe):
    def __init__(self):
        super(FCNETRecipe, self).__init__(
            name=BLACKBOX_NAME,
            cite_reference="Tabular benchmarks for joint architecture and hyperparameter optimization. "
            "Klein, A. and Hutter, F. 2019.",
        )

    def _generate_on_disk(self):
        generate_fcnet()


if __name__ == "__main__":
    FCNETRecipe().generate()

    # plot_learning_curves()
