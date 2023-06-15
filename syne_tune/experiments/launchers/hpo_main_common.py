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
import copy
import logging
from argparse import ArgumentParser, ArgumentTypeError
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable

from syne_tune.experiments.benchmark_definitions.common import BenchmarkDefinition

try:
    from coolname import generate_slug
except ImportError:
    print("coolname is not installed, will not be used")


DictStrKey = Dict[str, Any]


MapMethodArgsType = Callable[["ConfigDict", str, DictStrKey], DictStrKey]


ExtraArgsType = List[DictStrKey]


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


@dataclass
class Parameter:
    name: str
    type: Any
    help: str
    default: Any
    required: bool = False


class ConfigDict:
    """
    Dictinary with arguments for launcher scripts.
    Expected params as Parameter(name, type, default value)
    """

    _config: DictStrKey = {}
    __base_cl_parameters: List[Parameter] = [
        Parameter(
            name="experiment_tag",
            type=str,
            default=None,
            help="Tag for Syne-Tune experiments",
        ),
        Parameter(
            name="num_seeds",
            type=int,
            default=1,
            help="Number of seeds to run",
        ),
        Parameter(
            name="start_seed",
            type=int,
            default=0,
            help="First seed to run",
        ),
        Parameter(name="method", type=str, default=None, help="HPO method to run"),
        Parameter(
            name="save_tuner",
            type=str2bool,
            default=False,
            help="Serialize Tuner object at the end of tuning?",
        ),
        Parameter(
            name="n_workers",
            type=int,
            default=None,
            help="Number of workers (overwrites default of benchmark)",
        ),
        Parameter(
            name="max_wallclock_time",
            type=int,
            default=None,
            help="Maximum runtime for experiment (overwrites default of benchmark)",
        ),
        Parameter(
            name="random_seed",
            type=int,
            default=None,
            help="Master random seed (drawn at random if not given)",
        ),
        Parameter(
            name="max_size_data_for_model",
            type=int,
            default=None,
            help=f"Limits number of datapoints for surrogate model of BO, MOBSTER or HyperTune",
        ),
        Parameter(
            name="scale_max_wallclock_time",
            type=str2bool,
            default=False,
            help=(
                "If 1, benchmark.max_wallclock_time is multiplied by B / min(A, B),"
                "where A = n_workers and B = benchmark.n_workers"
            ),
        ),
        Parameter(
            name="use_long_tuner_name_prefix",
            type=str2bool,
            default=True,
            help=f"Use descriptive tuner name prefix for storing results",
        ),
        Parameter(
            name="launched_remotely",
            type=str2bool,
            default=False,
            help=f"Internal argument, do not use",
        ),
        Parameter(
            name="estimator_fit_backoff_wait_time",
            type=int,
            default=360,
            help=(
                "When a SageMaker training job fails with ResourceLimitExceeded, "
                "we try again after this many seconds. Pass 0 to switch this off. "
                "Only for launch_remote, not for hpo_main"
            ),
        ),
    ]
    __base_parameters = __base_cl_parameters + [
        Parameter(
            name="seeds",
            type=list,
            default=None,
            help="Seeds for this experiment, will be filled automatically based on start_seed and num_seeds",
        )
    ]
    __base_parameters_set = {item.name for item in __base_parameters}

    def __init__(self, **kwargs):
        self._config.update(**kwargs)
        if self.experiment_tag is None:
            try:
                self.experiment_tag = generate_slug(2)
            except Exception:
                self.experiment_tag = "syne-tune-experiment"

        self.seeds = list(range(self.start_seed, self.num_seeds))

    def __getattr__(self, attr):
        return self._config[attr]

    def __setattr__(self, attr, value):
        self._config[attr] = value

    def check_if_all_paremeters_present(self, desired_parameters: List[DictStrKey]):
        """
        Verify that all the parameers present in desired_parameters can be found in this ConfigDict
        """
        for dparam in desired_parameters:
            assert (
                dparam["name"] in self._config
            ), f"{dparam['name']} must be specified in the configuration for this experiment"

    def extra_parameters(self) -> List[DictStrKey]:
        """
        Return all parameters beyond those required
        Required are the defauls and those requested in argparse
        """
        return [
            {"name": name, "value": value}
            for name, value in self._config.items()
            if name not in self.__base_parameters_set
        ]

    def expand_base_arguments(self, extra_base_arguments: ExtraArgsType):
        """
        Expand the list of base argument for this experiment with those in extra_base_arguments
        """
        for extra_param in extra_base_arguments:
            if extra_param["name"] in self.__base_parameters:
                continue

            self.__base_parameters.append(
                Parameter(
                    name=extra_param["name"],
                    type=extra_param["type"],
                    default=extra_param.get("default", None),
                    help=extra_param.get("help", None),
                )
            )
            self.__base_parameters_set.add(extra_param["name"])

    @staticmethod
    def from_argparse(
        extra_args: Optional[ExtraArgsType] = None,
    ) -> "ConfigDict":
        """
        Build the configuration dict from command line arguments

        :param extra_args: Extra arguments for command line parser. Optional
        """
        parser = ArgumentParser(
            description=(
                "Run Syne Tune experiments for several HPO methods, benchmarks, "
                "and seeds (repetitions). Use hpo_main.py to launch experiments "
                "locally, or launch_remote.py to launch experiments remotely on AWS"
            ),
            epilog=(
                "For more information, please visit:\n"
                "https://syne-tune.readthedocs.io/en/latest/tutorials/benchmarking/README.html"
            ),
        )

        for param in ConfigDict.__base_cl_parameters:
            parser.add_argument(
                f"--{param.name}",
                type=param.type,
                default=param.default,
                required=param.required,
            )

        if extra_args is not None:
            local_extra_args = copy.deepcopy(extra_args)
            for kwargs in local_extra_args:
                name = kwargs.pop("name")
                assert (
                    name[0] != "-"
                ), f"Name entry '{name}' in extra_args invalid: No leading '-'"
                parser.add_argument("--" + name, **kwargs)

        known_args = parser.parse_args()
        config = ConfigDict(**vars(known_args))
        return config

    @staticmethod
    def from_dict(loaded_config: Dict = None) -> "ConfigDict":
        """
        Read the config from a dictionary
        """
        required_params = [
            item for item in ConfigDict.__base_parameters if item.required
        ]
        for required_param in required_params:
            assert (
                required_param.name in loaded_config
            ), f"{required_param} must be provided as part of configuration"

        final_config = {
            item.name: item.default for item in ConfigDict.__base_parameters
        }
        for key in loaded_config:
            final_config[key] = loaded_config[key]

        return ConfigDict(**final_config)


def set_logging_level(args: ConfigDict):
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger("syne_tune.optimizer.schedulers").setLevel(logging.WARNING)
        logging.getLogger("syne_tune.backend").setLevel(logging.WARNING)
        logging.getLogger(
            "syne_tune.backend.simulator_backend.simulator_backend"
        ).setLevel(logging.WARNING)


def get_metadata(
    seed: int,
    method: str,
    experiment_tag: str,
    benchmark_name: str,
    random_seed: int,
    max_size_data_for_model: Optional[int] = None,
    benchmark: Optional[BenchmarkDefinition] = None,
    extra_metadata: Optional[DictStrKey] = None,
) -> Dict[str, Any]:
    """Returns default value for ``metadata`` passed to :class:`~syne_tune.Tuner`.

    :param seed: Seed of repetition
    :param method: Name of method
    :param experiment_tag: Tag of experiment
    :param benchmark_name: Name of benchmark
    :param random_seed: Master random seed
    :param max_size_data_for_model: Limits number of datapoints for surrogate
        model of BO, MOBSTER or HyperTune
    :param benchmark: Optional. Take ``n_workers``, ``max_wallclock_time``
        from there
    :param extra_metadata: ``metadata`` updated by these at the end. Optional
    :return: Default ``metadata`` dictionary
    """
    metadata = {
        "seed": seed,
        "algorithm": method,
        "tag": experiment_tag,
        "benchmark": benchmark_name,
        "random_seed": random_seed,
    }
    if max_size_data_for_model is not None:
        metadata["max_size_data_for_model"] = max_size_data_for_model
    if benchmark is not None:
        metadata.update(
            {
                "n_workers": benchmark.n_workers,
                "max_wallclock_time": benchmark.max_wallclock_time,
            }
        )
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    return metadata


def extra_metadata(args, extra_args: ExtraArgsType) -> DictStrKey:
    result = dict()
    for extra_arg in extra_args:
        name = extra_arg["name"]
        value = getattr(args, name)
        if value is not None:
            result[name] = value
    return result


def config_from_argparse(
    extra_args: Optional[ExtraArgsType], backend_specific_args: ExtraArgsType
) -> ConfigDict:
    """
    Define the configuration directory based on extra arguments
    """
    if extra_args is None:
        extra_args = []
    else:
        extra_args = extra_args.copy()
    configuration = ConfigDict.from_argparse(
        extra_args=extra_args + backend_specific_args
    )
    return configuration
