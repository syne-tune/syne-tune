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
from typing import Optional, List, Dict, Any
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
from syne_tune.blackbox_repository.conversion_scripts.utils import (
    repository_path,
    blackbox_local_path,
)
from syne_tune.blackbox_repository.serialize import (
    serialize_metadata,
)
import syne_tune.config_space as cs
from syne_tune.blackbox_repository.blackbox import Blackbox
from syne_tune.constants import ST_WORKER_ITER
from syne_tune.util import is_increasing, is_positive_integer

import ConfigSpace
from yahpo_gym.benchmark_set import BenchmarkSet
from yahpo_gym.configuration import list_scenarios
from yahpo_gym import local_config

logger = logging.getLogger(__name__)


def download(target_path: Path, version: str):
    import urllib

    root = "https://github.com/slds-lmu/yahpo_data/archive/refs/tags/"

    target_file = target_path / f"yahpo_data-{version}"
    if not target_file.exists():
        logger.info(f"File {target_file} not found redownloading it.")
        urllib.request.urlretrieve(root + f"v{version}.zip", str(target_path) + ".zip")
        with zipfile.ZipFile(str(target_path) + ".zip", "r") as zip_ref:
            zip_ref.extractall(target_path)
    else:
        logger.info(f"File {target_file} found, skipping download.")


def _check_whether_iaml(benchmark: BenchmarkSet) -> bool:
    return benchmark.config.config_id.startswith("iaml_")


def _check_whether_rbv2(benchmark: BenchmarkSet) -> bool:
    return benchmark.config.config_id.startswith("rbv2_")


def _check_whether_nb301(benchmark: BenchmarkSet) -> bool:
    return benchmark.config.config_id == "nb301"


NB301_ATTRIBUTE_NAME_PREFIX = "NetworkSelectorDatasetInfo_COLON_darts_COLON_"


class BlackBoxYAHPO(Blackbox):
    """
    A wrapper that allows putting a 'YAHPO' BenchmarkInstance into a Blackbox.

    If ``fidelities`` is given, it restricts ``fidelity_values`` to these values.
    The sequence must be positive int and increasing. This works only if there
    is a single fidelity attribute with integer values (but note that for
    some specific YAHPO benchmarks, a fractional fidelity is transformed to
    an integer one).

    Even though YAHPO interpolates between fidelities, it can make sense
    to restrict them to the values which have really been acquired in the
    data. Note that this restricts multi-fidelity schedulers like
    :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`, in that all
    their rungs levels have to be fidelity values.

    For example, for YAHPO ``iaml``, the fidelity ``trainsize`` has been
    acquired at [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1], this is transformed
    to [1, 2, 4, 8, 12, 16, 20]. By default, the fidelity is
    represented by ``cs.randint(1, 20)``, but if ``fidelities`` is passed,
    it uses ``cs.ordinal(fidelities)``.

    :param benchmark: YAHPO ``BenchmarkSet``
    :param fidelities: See above
    """

    def __init__(
        self,
        benchmark: BenchmarkSet,
        fidelities: Optional[List[int]] = None,
    ):
        self.benchmark = benchmark
        super(BlackBoxYAHPO, self).__init__(
            configuration_space=cs_to_synetune(
                self.benchmark.get_opt_space(drop_fidelity_params=True)
            ),
            fidelity_space=cs_to_synetune(self.benchmark.get_fidelity_space()),
            objectives_names=self.benchmark.config.y_names,
        )
        self.num_seeds = 1
        self._is_iaml = _check_whether_iaml(benchmark)
        self._is_rbv2 = _check_whether_rbv2(benchmark)
        self._is_nb301 = _check_whether_nb301(benchmark)
        if self._is_rbv2:
            self.configuration_space["repl"] = 10
        self._shortened_keys = None
        self._initialize_for_scenario()
        # Has to be called after ``_initialize_for_scenario``, in order to
        # transform fidelity space for some of the YAHPO scenarios
        self._adjust_fidelity_space(fidelities)
        self._fidelity_multiplier = 0.05 if self._is_iaml or self._is_rbv2 else 1

    def _initialize_for_scenario(self):
        if self._is_iaml or self._is_rbv2:
            # For ``iaml_``, the fidelity ``trainsize`` has been evaluated at values
            # [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]. We multiply these values by 20
            # in order to obtain integers: [1, 2, 4, 8, 12, 16, 20]
            # For ``rbv2_``, the fidelity ``trainsize`` has been evaluated at values
            # [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]. We
            # multiply these values by 20 in order to obtain integers:
            # [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20]
            if self._is_iaml:
                assert len(self.fidelity_space) == 1
            domain = self.fidelity_space.get("trainsize")
            assert domain is not None
            assert isinstance(domain, cs.Float)
            assert domain.upper == 1 and domain.lower <= 0.05
            self.fidelity_space["trainsize"] = cs.randint(1, 20)
            if self._is_rbv2:
                # For ``rbv2_``, a second fidelity is ``repl``, but it is constant
                # 10, so can be removed
                assert len(self.fidelity_space) == 2
                assert "repl" in self.fidelity_space
                del self.fidelity_space["repl"]
        elif self._is_nb301:
            # Shorten overly long attribute names by removing the
            # common prefix
            len_prefix = len(NB301_ATTRIBUTE_NAME_PREFIX)
            shortened_keys = []

            def map_key(k: str) -> str:
                if k.startswith(NB301_ATTRIBUTE_NAME_PREFIX):
                    new_key = k[len_prefix:]
                    shortened_keys.append(new_key)
                    return new_key
                else:
                    return k

            self.configuration_space = {
                map_key(k): v for k, v in self.configuration_space.items()
            }
            self._shortened_keys = set(shortened_keys)

    def _adjust_fidelity_space(self, fidelities: Optional[List[int]]):
        assert len(self.fidelity_space) == 1, "Only one fidelity is supported"
        self._fidelity_name, domain = next(iter(self.fidelity_space.items()))
        assert (
            domain.value_type == int
        ), f"value_type of fidelity attribute must be int, but is {domain.value_type}"
        if fidelities is None:
            self._fidelity_values = np.arange(domain.lower, domain.upper + 1)
        else:
            assert is_increasing(fidelities) and is_positive_integer(
                fidelities
            ), f"fidelities = {fidelities} must be strictly increasing positive integers"
            assert (
                domain.lower <= fidelities[0] and fidelities[-1] <= domain.upper
            ), f"fidelities = {fidelities} must lie in [{domain.lower}, {domain.upper}]"
            self._fidelity_values = np.array(fidelities)
            self.fidelity_space[self._fidelity_name] = cs.ordinal(fidelities.copy())

    def _map_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if self._is_nb301:

            def map_key(k: str) -> str:
                if k in self._shortened_keys:
                    return NB301_ATTRIBUTE_NAME_PREFIX + k
                else:
                    return k

            return {map_key(k): v for k, v in config.items()}
        else:
            return config

    def _prepare_yahpo_configuration(
        self, configuration: Dict[str, Any], fidelity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Some of the hyperparameters are only active for certain values of other
        hyperparameters. We filter out the inactive ones, and add the fidelity to the
        configuration in order to interface with YAHPO.
        """
        configuration.update(fidelity)

        active_hyperparameters = self.benchmark.config_space.get_active_hyperparameters(
            ConfigSpace.Configuration(
                self.benchmark.config_space,
                values=configuration,
                allow_inactive_with_values=True,
            )
        )
        return {k: v for k, v in configuration.items() if k in active_hyperparameters}

    def _parse_fidelity(self, fidelity: Dict[str, Any]) -> Dict[str, Any]:
        if self._is_iaml or self._is_rbv2:
            k = "trainsize"
            fidelity_value = fidelity.get(k)
            assert (
                fidelity_value is not None
            ), f"fidelity = {fidelity} must contain key '{k}'"
            assert (
                fidelity_value in self.fidelity_values
            ), f"fidelity = {fidelity_value} not contained in {self.fidelity_values}"
            fidelity = {k: fidelity_value * self._fidelity_multiplier}
        return fidelity

    def _objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        configuration = self._map_configuration(configuration.copy())

        if fidelity is not None:
            configuration = self._prepare_yahpo_configuration(
                configuration, self._parse_fidelity(fidelity)
            )
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
            num_fidelities = self.fidelity_values.size
            num_objectives = len(self.objectives_names)
            result = np.empty((num_fidelities, num_objectives))
            configs = [
                self._prepare_yahpo_configuration(
                    configuration,
                    {self._fidelity_name: fidelity * self._fidelity_multiplier},
                )
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
        return self._fidelity_values

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


def instantiate_yahpo(
    scenario: str,
    check: bool = False,
    fidelities: Optional[List[int]] = None,
):
    """
    Instantiates a dict of ``BlackBoxYAHPO``, one entry for each instance.

    :param scenario:
    :param check: If False, ``objective_function`` of the blackbox does not
        check whether the input configuration is valid. This is faster, but
        calls fail silently if configurations are invalid.
    :return:
    """
    prefix = "yahpo-"
    assert scenario.startswith(prefix)
    scenario = scenario[len(prefix) :]

    # Note: Yahpo expects to see tasks such as "rbv2_xgb" with specific folders under the data-path.
    # for this reason, we create all blackboxes under a subdir yahpo/ to avoid name clashes with other blackboxes
    # such as "fcnet" or "lcbench".
    local_config.init_config()
    local_config.set_data_path(str(repository_path / "yahpo"))

    # Select a Benchmark, active_session False because the ONNX session can not be serialized.
    bench = benchmark_set.BenchmarkSet(scenario, active_session=False)

    return {
        instance: BlackBoxYAHPO(
            BenchmarkSet(
                scenario, active_session=False, instance=instance, check=check
            ),
            fidelities=fidelities,
        )
        for instance in bench.instances
    }


def serialize_yahpo(scenario: str, target_path: Path, version: str = "1.0"):
    assert scenario.startswith("yahpo-")
    scenario = scenario[6:]

    # download yahpo metadata and surrogate
    download(version=version, target_path=repository_path)

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
        assert name.startswith("yahpo-")
        self.scenario = name
        super(YAHPORecipe, self).__init__(
            name=name,
            cite_reference="YAHPO Gym - An Efficient Multi-Objective Multi-Fidelity Benchmark for Hyperparameter Optimization. "
            "Pfisterer F., Schneider S., Moosbauer J., Binder M., Bischl B., 2022",
        )

    def _generate_on_disk(self):
        # Note: Yahpo expects to see tasks such as "rbv2_xgb" with specific folders under the data-path.
        # for this reason, we create all blackboxes under a subdir yahpo/ to avoid name clashes with other blackboxes
        serialize_yahpo(
            self.scenario, target_path=blackbox_local_path(name=self.scenario)
        )


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
