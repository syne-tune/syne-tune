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
import os
import pandas as pd
import numpy as np
import zipfile
from pathlib import Path

from syne_tune.blackbox_repository.blackbox_offline import serialize
from syne_tune.blackbox_repository.conversion_scripts.blackbox_recipe import (
    BlackboxRecipe,
)
from syne_tune.blackbox_repository.conversion_scripts.utils import repository_path
from syne_tune.blackbox_repository.serialize import (
    serialize_metadata,
)
import syne_tune.config_space as cs
from syne_tune.blackbox_repository.blackbox import Blackbox
from typing import Dict, Optional, Callable, List, Tuple, Union


import ConfigSpace
from yahpo_gym.benchmark_set import BenchmarkSet
from yahpo_gym.configuration import list_scenarios
from yahpo_gym import local_config


def download(blackbox: str, version: str):
    import urllib

    root = "https://github.com/slds-lmu/yahpo_data/archive/refs/tags/"
    target = repository_path / f"{blackbox}"

    urllib.request.urlretrieve(root + f"v{version}.zip", str(target) + ".zip")

    with zipfile.ZipFile(str(target) + ".zip", "r") as zip_ref:
        zip_ref.extractall(target)

    # FIXME: Remove zip?
    # os.remove(str(target)+".zip")


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

    def _objective_function(
        self,
        configuration: Dict,
        fidelity: Optional[Dict] = None,
        seed: Optional[int] = None,
    ) -> Dict:
        if fidelity is not None:
            configuration.update(fidelity)
            return self.benchmark.objective_function(configuration, seed)[0]
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
            configs = []
            for fidelity in self.fidelity_values:
                config_with_fidelity = configuration.copy()
                config_with_fidelity[fidelity_name] = fidelity
                configs.append(config_with_fidelity)
            result_dicts = self.benchmark.objective_function(configs, seed=seed)

            for i, fidelity in enumerate(self.fidelity_values):
                result[i] = [
                    result_dicts[i][objective] for objective in self.objectives_names
                ]

            return result

    def set_instance(self, instance):
        """
        Set an instance for the underlying YAHPO Benchmark.
        """
        # Set the instance in the benchmark
        self.benchmark.set_instance(instance)
        # Update the configspace with the fixed instance
        if self.config.instance_names is not None:
            self.configuration_space[self.benchmark.config.instance_name] = cs.choice(
                [instance]
            )
        return self

    @property
    def instances(self) -> np.array:
        return self.benchmark.instances

    @property
    def fidelity_values(self) -> np.array:
        fids = next(iter(self.fidelity_space.values()))
        return np.arange(fids.lower, fids.upper)

    @property
    def time_attribute(self) -> str:
        """Name of the time column"""
        return self.benchmark.config.runtime_name


def cs_to_synetune(config_space):
    """
    Convert ConfigSpace.ConfigSpace to a synetune configspace.

    This should probably either be improved or we could also dump the 'synetune-configspaces' in 'yahpo_data'
    like we did for R (paradox) configspaces.
    Currently this does not cover all possible hyperparameters I think.
    """
    hps = config_space.get_hyperparameters()

    keys = []
    vals = []
    for a in hps:
        keys += [a.name]
        if isinstance(a, ConfigSpace.hyperparameters.CategoricalHyperparameter):
            vals += [cs.choice(a.choices)]
        if isinstance(a, ConfigSpace.hyperparameters.Constant):
            vals += [cs.choice([a.value])]
        if isinstance(a, ConfigSpace.hyperparameters.UniformIntegerHyperparameter):
            if a.log:
                vals += [cs.lograndint(a.lower, a.upper)]
            else:
                vals += [cs.randint(a.lower, a.upper)]
        if isinstance(a, ConfigSpace.hyperparameters.UniformFloatHyperparameter):
            if a.log:
                vals += [cs.loguniform(a.lower, a.upper)]
            else:
                vals += [cs.uniform(a.lower, a.upper)]

    # FIXME: This should also handle dependencies between hyperparameters.
    return dict(zip(keys, vals))


def instantiate_yahpo(scenario: str):

    # If the string starts with "YAHPO-" get rid of it
    if "YAHPO-" in scenario:
        scenario = scenario[6:]

    # FIXME: I can now create one scenario per instance but this is probably
    #        overkill since its just a copy of the same thing with a different constant set?
    return {
        instance: BlackBoxYAHPO(
            BenchmarkSet(scenario, active_session=False)
        ).set_instance(instance)
        for instance in b.instances
    }


def serialize_yahpo(scenario: str, version: str = "1.0"):
    """
    Serialize YAHPO (Metadata only for now)
    """
    # Step 1: Download Metadata
    download(blackbox="YAHPO", version=version)

    # Step 2: Set path in the yahpo config
    data_path = repository_path / "YAHPO" / f"yahpo_data-{version}"
    local_config.init_config()
    local_config.set_data_path(data_path)

    # Check if instances can be instantiated
    insts = instantiate_yahpo(scenario)

    # For now we only serialize metadata because everything else can be
    # obtained from YAHPO.
    path = Path(repository_path / "YAHPO")
    path.mkdir(exist_ok=True)
    serialize_metadata(
        path=path, metadata={"instances": list(insts.keys()), "data_path": data_path}
    )
    # FIXME: Do we need to serialize the ConfigSpace/ONNXSurrogate as well?
    # Might be required if the full benchmark should e.g. be distributed via S3?


class YAHPORecipe(BlackboxRecipe):
    def __init__(self, scenario: str):
        self.scenario = scenario
        super(YAHPORecipe, self).__init__(
            name=f"YAHPO-{name}",
            cite_reference="YAHPO Gym - An Efficient Multi-Objective Multi-Fidelity Benchmark for Hyperparameter Optimization"
            "Pfisterer F., Schneider S., Moosbauer J., Binder M., Bischl B., 2022",
        )

    def _generate_on_disk(self):
        serialize_yahpo(self.scenario)


yahpo_scenarios = [list_scenarios()]


if __name__ == "__main__":
    YAHPORecipe("lcbench").generate()
