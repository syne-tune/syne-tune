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
# Generating hp evaluations
import os

import pandas as pd
import numpy as np

from yahpo_gym import BenchmarkSet, local_config

from syne_tune.blackbox_repository.conversion_scripts.utils import (
    repository_path,
    blackbox_local_path,
)
from syne_tune.blackbox_repository.conversion_scripts.scripts.yahpo_import import (
    serialize_yahpo,
)

hp_names = {
    "rbv2_svm": [
        "cost",
        "kernel",
        "num.impute.selected.cpo",
        "tolerance",
        "gamma",
        "degree",
    ],
    "rbv2_aknn": [
        "M",
        "distance",
        "ef",
        "ef_construction",
        "k",
        "num.impute.selected.cpo",
    ],
    "rbv2_ranger": [
        "num.trees",
        "sample.fraction",
        "mtry.power",
        "respect.unordered.factors",
        "min.node.size",
        "splitrule",
        "num.impute.selected.cpo",
    ],
    "rbv2_glmnet": ["alpha", "s", "num.impute.selected.cpo"],
    "lcbench": [
        "batch_size",
        "learning_rate",
        "momentum",
        "weight_decay",
        "num_layers",
        "max_units",
        "max_dropout",
    ],
}

# Table 5 in the paper has surrogate quality, we use the ones has good quality
# https://arxiv.org/pdf/2109.03670.pdf
# For the rbv2 benchmarks, refer to https://rdrr.io/cran/kernlab/man/ksvm.html


def hp_values_hash(scenario, hp_dict):
    return {hp_name: hp_dict.get(hp_name, None) for hp_name in hp_names[scenario]}


def get_rbv_result(scenario, benchmark, hp, trainsize):
    hp["trainsize"] = trainsize

    f1_list = []
    auc_list = []
    acc_list = []
    for repl in range(1, 11):
        hp["repl"] = repl
        metric_dict = benchmark.objective_function(hp)[0]
        acc_list.append(metric_dict["acc"])
        f1_list.append(metric_dict["f1"])
        auc_list.append(metric_dict["auc"])

    hp_key = hp_values_hash(scenario, hp)

    return {
        "hp_key": hp_key,
        "train_frac": trainsize,
        "f1": np.mean(f1_list),
        "f1_std": np.std(f1_list),
        "acc": np.mean(acc_list),
        "acc_std": np.std(acc_list),
        "auc": np.mean(auc_list),
        "auc_std": np.std(auc_list),
    }


scenarios = ["rbv2_svm", "rbv2_aknn", "rbv2_ranger", "rbv2_glmnet"]


def run():
    for scenario in scenarios:

        b = BenchmarkSet(scenario=scenario)
        print(b.targets)
        print(b.instances)

        config_space = b.get_opt_space(drop_fidelity_params=True)
        config_space.seed(666)
        hps = config_space.sample_configuration(3000)

        for instance in b.instances[:10]:
            b.set_instance(instance)

            results = []
            for hp in hps:
                hp = hp.get_dictionary()

                # Evaluate the configurattion
                for trainsize in [0.05, 0.25, 0.5, 0.75, 1.0]:
                    result = get_rbv_result(scenario, b, hp, trainsize)
                    results.append(result)

            if not os.path.exists("yahpo_data"):
                os.makedirs("yahpo_data")
            if not os.path.exists("yahpo_data/" + scenario):
                os.makedirs("yahpo_data/" + scenario)

            pd.DataFrame(results).to_csv(
                f"yahpo_data/{scenario}/{instance}.csv", index=False
            )


if __name__ == "__main__":
    local_config.init_config()
    local_config.set_data_path(str(repository_path / "yahpo"))
    for scenario in scenarios:
        # Use syne-tune to download yahpo data
        scenario_long = "yahpo-" + scenario
        serialize_yahpo(
            scenario_long, target_path=blackbox_local_path(name=scenario_long)
        )
    run()
