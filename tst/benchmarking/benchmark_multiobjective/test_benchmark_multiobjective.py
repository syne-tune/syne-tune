import unittest
from unittest.mock import MagicMock, patch
from benchmarking.commons.default_baselines import RandomSearch
from benchmarking.nursery.benchmark_multiobjective.baselines import (
    Methods,
    MOREABench,
    LSOBOBench,
)
from benchmarking.nursery.benchmark_multiobjective.benchmark_definitions import (
    benchmark_definitions,
    nas201_mo_benchmark,
)
from benchmarking.nursery.benchmark_multiobjective.hpo_main import main


class MultiObjectiveTests(unittest.TestCase):
    @patch(
        "benchmarking.nursery.benchmark_multiobjective.hpo_main.run_experiment",
        new_callable=MagicMock,
    )
    @patch("benchmarking.nursery.benchmark_multiobjective.hpo_main.parse_args")
    def test_run_experiment_fn_is_called_the_expected_number_of_times(
        self, mock_parse_args, mock_run_experiment
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

        mock_args = unittest.mock.MagicMock()
        mock_args.experiment_tag = "test-experiment-tag"
        mock_parse_args.return_value = (
            mock_args,
            methods,
            benchmark_definitions,
            [1, 2],
        )

        main(methods, benchmark_definitions)

        assert mock_run_experiment.call_count == 18
