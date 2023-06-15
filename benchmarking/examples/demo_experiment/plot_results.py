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
from typing import Dict, Any, Optional, List, Set
import logging

from baselines import methods
from benchmark_definitions import benchmark_definitions
from hpo_main import RungLevelsExtraResults
from syne_tune.experiments import ComparativeResults, PlotParameters, SubplotParameters


def metadata_to_setup(metadata: Dict[str, Any]) -> Optional[str]:
    # The setup is the algorithm. No filtering
    return metadata["algorithm"]


SETUP_TO_SUBPLOT = {
    "ASHA": 0,
    "MOBSTER": 0,
    "ASHA-TANH": 1,
    "MOBSTER-TANH": 1,
    "ASHA-RELU": 2,
    "MOBSTER-RELU": 2,
    "RS": 3,
    "BO": 3,
}


def metadata_to_subplot(metadata: Dict[str, Any]) -> Optional[int]:
    return SETUP_TO_SUBPLOT[metadata["algorithm"]]


def _print_extra_results(
    extra_results: Dict[str, Dict[str, List[float]]],
    keys: List[str],
    skip_setups: Set[str],
):
    for setup_name, results_for_setup in extra_results.items():
        if setup_name not in skip_setups:
            print(f"[{setup_name}]:")
            for key in keys:
                values = results_for_setup[key]
                print(f"  {key}: {[int(x) for x in values]}")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    experiment_name = "docs-2"
    experiment_names = (experiment_name,)
    setups = list(methods.keys())
    num_runs = 20
    download_from_s3 = False  # Set ``True`` in order to download files from S3
    # Plot parameters across all benchmarks
    plot_params = PlotParameters(
        xlabel="wall-clock time",
        aggregate_mode="iqm_bootstrap",
        grid=True,
    )
    # We would like four subplots (2 row, 2 columns), each showing two setups.
    # Each subplot gets its own title, and legends are shown in each,
    plot_params.subplots = SubplotParameters(
        nrows=2,
        ncols=2,
        kwargs=dict(sharex="all", sharey="all"),
        titles=[
            "activations tuned",
            "activations = tanh",
            "activations = relu",
            "single fidelity",
        ],
        title_each_figure=True,
        legend_no=[0, 1, 2, 3],
    )

    # The creation of ``results`` downloads files from S3 (only if
    # ``download_from_s3 == True``), reads the metadata and creates an inverse
    # index. If any result files are missing, or there are too many of them,
    # warning messages are printed
    results = ComparativeResults(
        experiment_names=experiment_names,
        setups=setups,
        num_runs=num_runs,
        metadata_to_setup=metadata_to_setup,
        plot_params=plot_params,
        metadata_to_subplot=metadata_to_subplot,
        download_from_s3=download_from_s3,
    )

    # We can now create plots for the different benchmarks:
    # - We store the figures as PNG files
    # - We also load the extra results collected during the experiments
    #   (recall that we monitored sizes of rungs for ASHA and MOBSTER).
    #   Instead of plotting their values over time, we print out their
    #   values at the end of each experiment
    extra_results_keys = RungLevelsExtraResults().keys()
    skip_setups = {"RS", "BO"}
    # First: fcnet-protein
    benchmark_name = "fcnet-protein"
    benchmark = benchmark_definitions[benchmark_name]
    # These parameters overwrite those given at construction
    plot_params = PlotParameters(
        metric=benchmark.metric,
        mode=benchmark.mode,
        ylim=(0.22, 0.30),
    )
    extra_results = results.plot(
        benchmark_name=benchmark_name,
        plot_params=plot_params,
        file_name=f"./{experiment_name}-{benchmark_name}.png",
        extra_results_keys=extra_results_keys,
    )["extra_results"]
    _print_extra_results(extra_results, extra_results_keys, skip_setups=skip_setups)
    # Next: fcnet-slice
    benchmark_name = "fcnet-slice"
    benchmark = benchmark_definitions[benchmark_name]
    # These parameters overwrite those given at construction
    plot_params = PlotParameters(
        metric=benchmark.metric,
        mode=benchmark.mode,
        ylim=(0.00025, 0.0012),
    )
    results.plot(
        benchmark_name=benchmark_name,
        plot_params=plot_params,
        file_name=f"./{experiment_name}-{benchmark_name}.png",
        extra_results_keys=extra_results_keys,
    )
    _print_extra_results(extra_results, extra_results_keys, skip_setups=skip_setups)
