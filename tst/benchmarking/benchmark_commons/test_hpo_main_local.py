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
import unittest
from unittest.mock import MagicMock, patch
from benchmarking.commons.default_baselines import RandomSearch
from benchmarking.commons.hpo_main_common import ConfigDict
from benchmarking.nursery.benchmark_multiobjective.baselines import (
    Methods,
    MOREABench,
    LSOBOBench,
)
from benchmarking.nursery.benchmark_multiobjective.benchmark_definitions import (
    nas201_mo_benchmark,
)
from benchmarking.nursery.benchmark_multiobjective.hpo_main import main


class HPOMainLocalTests(unittest.TestCase):
    @patch("benchmarking.commons.hpo_main_simulator.config_from_argparse", new_callable=MagicMock)
    @patch("benchmarking.commons.hpo_main_simulator.Tuner", new_callable=MagicMock)
    @patch("benchmarking.commons.hpo_main_simulator.BlackboxRepositoryBackend", new_callable=MagicMock)
    def test_experiment_is_run_the_expected_number_of_times(
        self, mock_blackbox_repository_backend, mock_tuner, mock_config_from_argparse
    ):
        methods = {
            Methods.RS: lambda method_arguments: RandomSearch(method_arguments),
            Methods.MOREA: lambda method_arguments: MOREABench(method_arguments),
            Methods.LSOBO: lambda method_arguments: LSOBOBench(method_arguments),
        }

        benchmark_definitions = {
            "nas201-cifar10": nas201_mo_benchmark("cifar10"),
            "nas201-cifar100": nas201_mo_benchmark("cifar100"),
            "nas201-ImageNet16-120": nas201_mo_benchmark("ImageNet16-120"),
        }

        seeds = [1, 2]

        mock_config_from_argparse.return_value = ConfigDict.from_dict(
            {'experiment_tag': 'my-new-experiment', 'num_seeds': 2, 'start_seed': 0, 'method': None, 'save_tuner': 0,
             'n_workers': None, 'max_wallclock_time': None, 'random_seed': None, 'max_size_data_for_model': None,
             'scale_max_wallclock_time': 0, 'use_long_tuner_name_prefix': True, 'launched_remotely': False,
             'benchmark': None, 'verbose': 0, 'support_checkpointing': 1, 'fcnet_ordinal': 'nn-log',
             'restrict_configurations': 0, 'seeds': [0, 1]})

        main(methods, benchmark_definitions)

        expected_call_count = len(methods) * len(benchmark_definitions) * len(seeds)
        assert mock_tuner.call_count == expected_call_count
