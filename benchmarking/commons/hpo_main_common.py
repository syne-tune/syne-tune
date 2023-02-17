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
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable

from benchmarking.commons.benchmark_definitions.common import BenchmarkDefinition
from syne_tune import Tuner

try:
    from coolname import generate_slug
except ImportError:
    print("coolname is not installed, will not be used")


DictStrKey = Dict[str, Any]


MapExtraArgsType = Callable[[Any, str, DictStrKey], DictStrKey]


ExtraArgsType = List[DictStrKey]


PostProcessingType = Callable[[Tuner], Any]


def str2bool(v):
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


class ConfigDict(dict):
    """
    Dictinary with arguments for benchmarking
    Expected params as Parameter(name, type, default value)
    """

    __base_parameters: List[Parameter] = [
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
            default=0,
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
            help=f"Limits number of datapoints for surrogate model of MOBSTER or HyperTune",
        ),
        Parameter(
            name="scale_max_wallclock_time",
            type=str2bool,
            default=0,
            help=(
                "If 1, benchmark.max_wallclock_time is multiplied by B / min(A, B),"
                "where A = n_workers and B = benchmark.n_workers"
            ),
        ),
    ]

    def __init__(self, *args, **parameters):
        super().__init__(*args, **parameters)
        if self.experiment_tag is None:
            try:
                self.experiment_tag = generate_slug(2)
            except Exception:
                self.experiment_tag = "syne-tune-experiment"

        self.seeds = list(range(self.start_seed, self.num_seeds))

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    @staticmethod
    def from_argparse(
        extra_args: Optional[ExtraArgsType] = None,
    ) -> "ConfigDict":
        """
        Build the configuration dict from command line arguments

        ``map_extra_args`` can be used to modify ``method_kwargs`` for constructing
        :class:`~benchmarking.commons.baselines.MethodArguments`, depending on
        ``args`` returned by :func:`parse_args` and the method. Its signature is
        :code:`method_kwargs = map_extra_args(args, method, method_kwargs)`, where
        ``method`` is the name of the baseline.

        :param extra_args: Extra arguments for command line parser. Optional
        :param map_extra_args: See above, optional
        """
        parser = ArgumentParser(
            description=(
                "Run Syne Tune experiments for several HPO methods, benchmarks, "
                "and seeds (repetitions). Use hpo_main.py to launch experiments "
                "locally, or launch_remote.py to launch experiments remotely on AWS"
            ),
            epilog="For more information, please visit:\nhttps://syne-tune.readthedocs.io/en/latest/tutorials/benchmarking/README.html",
        )

        for param in ConfigDict.__base_parameters:
            parser.add_argument(
                f"--{param.name}",
                type=param.type,
                default=param.default,
                required=param.required,
            )

        if extra_args is not None:
            extra_args = copy.deepcopy(extra_args)
            for kwargs in extra_args:
                name = kwargs.pop("name")
                assert (
                    name[0] != "-"
                ), f"Name entry '{name}' in extra_args invalid: No leading '-'"
                parser.add_argument("--" + name, **kwargs)

        known_args, extra_args = parser.parse_known_args()
        config = ConfigDict(**vars(known_args))
        return config

    @staticmethod
    def from_dict(loaded_config: Dict = None) -> "ConfigDict":
        """
        Read the config from a dictionary
        """
        required_params = [item for item in ConfigDict.__base_parameters if item.required]
        for required_param in required_params:
            assert required_param.name in loaded_config, f"{required_param} must be provided as part of configuration"

        final_config = {
            item.name: item.default for item in ConfigDict.__base_parameters
        }
        for key in loaded_config:
            final_config[key] = loaded_config[key]

        return ConfigDict(**final_config)


def parse_args(
    methods: DictStrKey, extra_args: Optional[ExtraArgsType] = None
) -> (Any, List[str], List[int]):
    """Default implementation for parsing command line arguments.

    :param methods: If ``--method`` is not given, then ``method_names`` are the
        keys of this dictionary
    :param extra_args: List of dictionaries, containing additional arguments
        to be passed. Must contain ``name`` for argument name (without leading
        ``"--"``), and other kwargs to ``parser.add_argument``. Optional
    :return: ``(args, method_names, seeds)``, where ``args`` is result of
        ``parser.parse_args()``, ``method_names`` see ``methods``, and
        ``seeds`` are list of seeds specified by ``--num_seeds`` and ``--start_seed``
    """
    try:
        default_experiment_tag = generate_slug(2)
    except Exception:
        default_experiment_tag = "syne-tune-experiment"
    parser = ArgumentParser(
        description=(
            "Run Syne Tune experiments for several HPO methods, benchmarks, "
            "and seeds (repetitions). Use hpo_main.py to launch experiments "
            "locally, or launch_remote.py to launch experiments remotely on AWS"
        ),
        epilog="For more information, please visit:\nhttps://syne-tune.readthedocs.io/en/latest/tutorials/benchmarking/README.html",
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default=default_experiment_tag,
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=1,
        help="Number of seeds to run",
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default=0,
        help="First seed to run",
    )
    parser.add_argument("--method", type=str, help="HPO method to run")
    parser.add_argument(
        "--save_tuner",
        type=int,
        default=0,
        help="Serialize Tuner object at the end of tuning?",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        help="Number of workers (overwrites default of benchmark)",
    )
    parser.add_argument(
        "--max_wallclock_time",
        type=int,
        help="Maximum runtime for experiment (overwrites default of benchmark)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Master random seed (drawn at random if not given)",
    )
    parser.add_argument(
        "--max_size_data_for_model",
        type=int,
        help=f"Limits number of datapoints for surrogate model of MOBSTER or HyperTune",
    )
    parser.add_argument(
        "--scale_max_wallclock_time",
        type=int,
        default=0,
        help=(
            "If 1, benchmark.max_wallclock_time is multiplied by B / min(A, B),"
            "where A = n_workers and B = benchmark.n_workers"
        ),
    )
    if extra_args is not None:
        extra_args = copy.deepcopy(extra_args)
        for kwargs in extra_args:
            name = kwargs.pop("name")
            assert (
                name[0] != "-"
            ), f"Name entry '{name}' in extra_args invalid: No leading '-'"
            parser.add_argument("--" + name, **kwargs)
    args = parser.parse_args()
    args.save_tuner = bool(args.save_tuner)
    args.scale_max_wallclock_time = bool(args.scale_max_wallclock_time)
    seeds = list(range(args.start_seed, args.num_seeds))
    method_names = [args.method] if args.method is not None else list(methods.keys())
    return args, method_names, seeds


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
        model of MOBSTER or HyperTune
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
