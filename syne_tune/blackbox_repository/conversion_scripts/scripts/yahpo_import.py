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
Wrap Surrogates from 
YAHPO Gym - An Efficient Multi-Objective Multi-Fidelity Benchmark for Hyperparameter Optimization
Florian Pfisterer, Lennart Schneider, Julia Moosbauer, Martin Binder, Bernd Bischl
"""
import logging
import shutil

from yahpo_gym import benchmark_set
import numpy as np
import zipfile
from pathlib import Path

from syne_tune.blackbox_repository.conversion_scripts.blackbox_recipe import (
    BlackboxRecipe,
)
from syne_tune.blackbox_repository.conversion_scripts.scripts import (
    default_metric,
    metric_elapsed_time,
    resource_attr,
)
from syne_tune.blackbox_repository.conversion_scripts.utils import repository_path
from syne_tune.blackbox_repository.serialize import (
    serialize_metadata,
)
import syne_tune.config_space as cs
from syne_tune.blackbox_repository.blackbox import Blackbox
from typing import Dict, Optional


import ConfigSpace
from yahpo_gym.benchmark_set import BenchmarkSet
from yahpo_gym.configuration import list_scenarios
from yahpo_gym import local_config

from syne_tune.constants import ST_WORKER_ITER


def download(target_path: Path, version: str):
    import urllib

    root = "https://github.com/slds-lmu/yahpo_data/archive/refs/tags/"

    target_file = target_path / f"yahpo_data-{version}"
    if not target_file.exists():
        logging.info(f"File {target_file} not found redownloading it.")
        urllib.request.urlretrieve(root + f"v{version}.zip", str(target_path) + ".zip")
        with zipfile.ZipFile(str(target_path) + ".zip", "r") as zip_ref:
            zip_ref.extractall(target_path)
    else:
        logging.info(f"File {target_file} found, skipping download.")


class BlackBoxYAHPO(Blackbox):
    """
    A wrapper that allows putting a 'YAHPO' BenchmarkInstance into a Blackbox.
    """

    def __init__(self, benchmark):
        self.benchmark = benchmark
        super(BlackBoxYAHPO, self).__init__(
            configuration_space=cs_to_synetune(
                self.benchmark.get_opt_space(drop_fidelity_params=True)
            ),
            fidelity_space=cs_to_synetune(self.benchmark.get_fidelity_space()),
            objectives_names=self.benchmark.config.y_names,
        )
        self.num_seeds = 1

    def _objective_function(
        self,
        configuration: Dict,
        fidelity: Optional[Dict] = None,
        seed: Optional[int] = None,
    ) -> Dict:
        configuration = configuration.copy()
        if fidelity is not None:
            configuration.update(fidelity)
            return self.benchmark.objective_function(configuration, seed=seed)[0]
        else:
            """
            copying the parent comment of the parent class:
            "not passing a fidelity is possible if either the blackbox does not have a fidelity space
            or if it has a single fidelity in its fidelity space. In the latter case, all fidelities are returned in form
            of a tensor with shape (num_fidelities, num_objectives)."
            This is used for efficiency (it is much faster to retrieve a full row in an array in term of read time).
            """
            # returns a tensor of shape (num_fidelities, num_objectives)
            num_fidelities = len(self.fidelity_values)
            num_objectives = len(self.objectives_names)
            result = np.empty((num_fidelities, num_objectives))
            fidelity_name = next(iter(self.fidelity_space.keys()))
            configs = [
                {**configuration, fidelity_name: fidelity}
                for fidelity in self.fidelity_values
            ]
            result_dicts = self.benchmark.objective_function(configs, seed=seed)

            for i, result_dict in enumerate(result_dicts):
                result[i] = [
                    result_dict[objective] for objective in self.objectives_names
                ]

            return result

    def set_instance(self, instance):
        """
        Set an instance for the underlying YAHPO Benchmark.
        """
        # Set the instance in the benchmark
        self.benchmark.set_instance(instance)
        # Update the configspace with the fixed instance
        if self.benchmark.config.instance_names:
            instance_names = self.benchmark.config.instance_names
        else:
            instance_names = "instance-names"
        self.configuration_space[instance_names] = instance
        return self

    @property
    def instances(self) -> np.array:
        return self.benchmark.instances

    @property
    def fidelity_values(self) -> np.array:
        fids = next(iter(self.fidelity_space.values()))
        return np.arange(fids.lower, fids.upper + 1)

    @property
    def time_attribute(self) -> str:
        """Name of the time column"""
        return self.benchmark.config.runtime_name


def cs_to_synetune(config_space):
    """
    Convert ConfigSpace.ConfigSpace to a synetune configspace.

    TODO cover all possible hyperparameters of ConfigSpace.ConfigSpace, right now we only convert the one we need.
    """
    hps = config_space.get_hyperparameters()

    keys = []
    vals = []
    for a in hps:
        keys += [a.name]
        if isinstance(a, ConfigSpace.hyperparameters.CategoricalHyperparameter):
            if len(a.choices) > 1:
                val = cs.choice(a.choices)
            else:
                val = a.choices[0]
            vals += [val]
        elif isinstance(a, ConfigSpace.hyperparameters.Constant):
            vals += [a.value]
        elif isinstance(a, ConfigSpace.hyperparameters.UniformIntegerHyperparameter):
            if a.log:
                vals += [cs.lograndint(a.lower, a.upper)]
            else:
                vals += [cs.randint(a.lower, a.upper)]
        elif isinstance(a, ConfigSpace.hyperparameters.UniformFloatHyperparameter):
            if a.log:
                vals += [cs.loguniform(a.lower, a.upper)]
            else:
                vals += [cs.uniform(a.lower, a.upper)]
        else:
            raise ValueError(
                f"Hyperparameter {a.name} has type {type(a)} which is not supported in this converter."
            )
    # FIXME: This should also handle dependencies between hyperparameters.
    return dict(zip(keys, vals))


def instantiate_yahpo(scenario: str, check: bool = False):
    """
    Instantiates a dict of `BlackBoxYAHPO`, one entry for each instance.

    :param scenario:
    :param check: If False, `objective_function` of the blackbox does not
        check whether the input configuration is valid. This is faster, but
        calls fail silently if configurations are invalid.
    :return:
    """
    prefix = "yahpo-"
    assert scenario.startswith(prefix)
    scenario = scenario[len(prefix) :]

    local_config.init_config()
    local_config.set_data_path(str(repository_path / "yahpo"))

    # Select a Benchmark, active_session False because the ONNX session can not be serialized.
    bench = benchmark_set.BenchmarkSet(scenario, active_session=False)

    return {
        instance: BlackBoxYAHPO(
            BenchmarkSet(scenario, active_session=False, instance=instance, check=check)
        )
        for instance in bench.instances
    }


def serialize_yahpo(scenario: str, version: str = "1.0"):
    """
    Serialize YAHPO (Metadata only for now)
    """
    assert scenario.startswith("yahpo-")
    scenario = scenario[6:]

    # download yahpo metadata and surrogate
    download(version=version, target_path=repository_path)

    target_path = repository_path / f"yahpo" / scenario

    # copy files to yahpo-scenario
    if target_path.exists():
        shutil.rmtree(target_path)
    shutil.copytree(
        str(repository_path / f"yahpo_data-{version}" / scenario), str(target_path)
    )

    # For now we only serialize metadata because everything else can be obtained from YAHPO.
    serialize_metadata(
        path=target_path,
        metadata={
            metric_elapsed_time: "time",
            default_metric: "val_accuracy",
            resource_attr: ST_WORKER_ITER,  # TODO, ressource not present, we can use ST_WORKER_ITER
        },
    )


class YAHPORecipe(BlackboxRecipe):
    def __init__(self, name: str):
        self.scenario = name
        super(YAHPORecipe, self).__init__(
            name=name,
            cite_reference="YAHPO Gym - An Efficient Multi-Objective Multi-Fidelity Benchmark for Hyperparameter Optimization. "
            "Pfisterer F., Schneider S., Moosbauer J., Binder M., Bischl B., 2022",
        )

    def _generate_on_disk(self):
        serialize_yahpo(self.scenario)


yahpo_scenarios = list_scenarios()


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    scenario = "lcbench"

    YAHPORecipe(f"yahpo-{scenario}").generate()

    # plot one learning-curve for sanity-check
    from syne_tune.blackbox_repository import load_blackbox

    bb_dict = load_blackbox(f"yahpo-{scenario}", skip_if_present=False)
    first_task = next(iter(bb_dict.keys()))
    b = bb_dict[first_task]
    configuration = {k: v.sample() for k, v in b.configuration_space.items()}
    errors = []
    runtime = []

    import matplotlib.pyplot as plt

    for i in range(1, 52):
        res = b.objective_function(configuration=configuration, fidelity={"epoch": i})
        errors.append(res["val_accuracy"])
        runtime.append(res["time"])

    plt.plot(np.cumsum(runtime), errors)
    plt.show()
