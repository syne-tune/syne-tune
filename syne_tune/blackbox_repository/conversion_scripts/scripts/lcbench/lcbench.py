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
import zipfile
import urllib

import pandas as pd
import numpy as np

from syne_tune.blackbox_repository.blackbox_tabular import serialize, BlackboxTabular
from syne_tune.blackbox_repository.conversion_scripts.blackbox_recipe import (
    BlackboxRecipe,
)
from syne_tune.blackbox_repository.conversion_scripts.scripts import (
    metric_elapsed_time,
    default_metric,
    resource_attr,
)
from syne_tune.config_space import randint, lograndint, uniform, loguniform
from syne_tune.util import catchtime
from syne_tune.blackbox_repository.conversion_scripts.scripts.lcbench.api import (
    Benchmark,
)
from syne_tune.blackbox_repository.conversion_scripts.utils import repository_path


BLACKBOX_NAME = "lcbench"

METRIC_ACCURACY = "val_accuracy"

METRIC_ELAPSED_TIME = "time"

RESOURCE_ATTR = "epoch"

MAX_RESOURCE_LEVEL = 52

CONFIGURATION_SPACE = {
    "num_layers": randint(1, 5),
    "max_units": lograndint(64, 1024),
    "batch_size": lograndint(16, 512),
    "learning_rate": loguniform(1e-4, 1e-1),
    "weight_decay": uniform(1e-5, 1e-1),
    "momentum": uniform(0.1, 0.99),
    "max_dropout": uniform(0.0, 1.0),
}


def convert_task(bench, dataset_name):
    n_config = 2000
    num_epochs = MAX_RESOURCE_LEVEL
    configs = [
        bench.query(dataset_name=dataset_name, tag="config", config_id=i)
        for i in range(n_config)
    ]
    hyperparameters = pd.DataFrame(configs)
    # remove constant columns
    hyperparameters = hyperparameters.loc[
        :, (hyperparameters != hyperparameters.iloc[0]).any()
    ]
    objective_evaluations = np.zeros((n_config, 1, num_epochs, 2))

    fidelity_space = {RESOURCE_ATTR: randint(lower=1, upper=num_epochs)}
    for j, tag in enumerate(["Train/val_accuracy", "time"]):
        for i in range(n_config):
            objective_evaluations[i, 0, :, j] = bench.query(
                dataset_name=dataset_name, tag=tag, config_id=i
            )
    return BlackboxTabular(
        hyperparameters=hyperparameters,
        configuration_space=CONFIGURATION_SPACE,
        fidelity_space=fidelity_space,
        objectives_evaluations=objective_evaluations,
        fidelity_values=np.arange(1, num_epochs + 1),
        objectives_names=[METRIC_ACCURACY, METRIC_ELAPSED_TIME],
    )


class LCBenchRecipe(BlackboxRecipe):
    def __init__(self):
        super(LCBenchRecipe, self).__init__(
            name=BLACKBOX_NAME,
            cite_reference="Auto-PyTorch: Multi-Fidelity MetaLearning for Efficient and Robust AutoDL. "
            "Lucas Zimmer, Marius Lindauer, Frank Hutter. 2020.",
        )

    def _generate_on_disk(self):
        data_file = repository_path / "data_2k_lw.zip"
        if not data_file.exists():
            src = "https://figshare.com/ndownloader/files/21188598"
            print(f"did not find {data_file}, downloading {src}")
            urllib.request.urlretrieve(src, data_file)

        with zipfile.ZipFile(data_file, "r") as zip_ref:
            zip_ref.extractall(repository_path)

        with catchtime("converting"):
            bench = Benchmark(str(repository_path / "data_2k_lw.json"), cache=False)
            bb_dict = {
                task: convert_task(bench, task) for task in bench.get_dataset_names()
            }

        with catchtime("saving to disk"):
            serialize(
                bb_dict=bb_dict,
                path=repository_path / self.name,
                metadata={
                    metric_elapsed_time: METRIC_ELAPSED_TIME,
                    default_metric: METRIC_ACCURACY,
                    resource_attr: RESOURCE_ATTR,
                },
            )


if __name__ == "__main__":
    LCBenchRecipe().generate()
