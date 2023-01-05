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
from argparse import ArgumentParser
from typing import Optional, List, Dict, Any

from benchmarking.commons.benchmark_definitions.common import BenchmarkDefinition

try:
    from coolname import generate_slug
except ImportError:
    print("coolname is not installed, will not be used")


def parse_args(
    methods: Dict[str, Any], extra_args: Optional[List[dict]] = None
) -> (Any, List[str], List[int]):
    """Default implementation for parsing command line arguments.

    :param methods: If ``--method`` is not given, then ``method_names`` are the
        keys of this dictionary
    :param extra_args: List of dictionaries, containing additional arguments
        to be passed. Must contain ``name`` for argument name (without leading
        ``"--"``), and other kwargs to ``parser.add_argument``. Optional
    :return: ``(args, method_names, seeds)``, where ``args`` is result of
        ``parser.parse_known_args()``, ``method_names`` see ``methods``, and
        ``seeds`` are list of seeds specified by ``--num_seeds`` and ``--start_seed``
    """
    try:
        default_experiment_tag = generate_slug(2)
    except Exception:
        default_experiment_tag = "syne-tune-experiment"
    parser = ArgumentParser()
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
    if extra_args is not None:
        extra_args = copy.deepcopy(extra_args)
        for kwargs in extra_args:
            name = kwargs.pop("name")
            parser.add_argument("--" + name, **kwargs)
    args, _ = parser.parse_known_args()
    args.save_tuner = bool(args.save_tuner)
    seeds = list(range(args.start_seed, args.num_seeds))
    method_names = [args.method] if args.method is not None else list(methods.keys())
    return args, method_names, seeds


def set_logging_level(args):
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
    benchmark: Optional[BenchmarkDefinition] = None,
    extra_args: Optional[dict] = None,
) -> Dict[str, Any]:
    """Returns default value for ``metadata`` passed to :class:`~syne_tune.Tuner`.

    :param seed: Seed of repetition
    :param method: Name of method
    :param experiment_tag: Tag of experiment
    :param benchmark_name: Name of benchmark
    :param random_seed: Master random seed
    :param benchmark: Optional. Take ``n_workers``, ``max_wallclock_time``
        from there
    :param extra_args: ``metadata`` updated by these at the end. Optional
    :return: Default ``metadata`` dictionary
    """
    metadata = {
        "seed": seed,
        "algorithm": method,
        "tag": experiment_tag,
        "benchmark": benchmark_name,
        "random_seed": random_seed,
    }
    if benchmark is not None:
        metadata.update(
            {
                "n_workers": benchmark.n_workers,
                "max_wallclock_time": benchmark.max_wallclock_time,
            }
        )
    if extra_args is not None:
        metadata.update(extra_args)
    return metadata
