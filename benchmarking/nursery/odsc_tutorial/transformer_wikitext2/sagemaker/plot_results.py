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
from typing import Dict, Any, Optional
import logging

from transformer_wikitext2.benchmark_definitions import benchmark_definitions
from syne_tune.experiments import ComparativeResults, PlotParameters, SubplotParameters


TMLR10_SETUPS = [
    "2 workers",
    "4 workers",
    "8 workers",
]


def metadata_to_setup(metadata: Dict[str, Any]) -> Optional[str]:
    return f"{metadata['n_workers']} workers"


TMLR10_METHOD_TO_SUBPLOT = {
    "RS": 0,
    "BO": 1,
    "ASHA": 2,
    "MOBSTER": 3,
}


def metadata_to_subplot(metadata: dict) -> Optional[int]:
    return TMLR10_METHOD_TO_SUBPLOT[metadata["algorithm"]]


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    experiment_names = ("tmlr-10",)
    num_runs = 5
    download_from_s3 = False  # Set ``True`` in order to download files from S3
    # Plot parameters across all benchmarks
    plot_params = PlotParameters(
        xlabel="wall-clock time",
        aggregate_mode="iqm_bootstrap",
        grid=True,
    )
    # We would like to have 4 subfigures, one for each method
    plot_params.subplots = SubplotParameters(
        nrows=2,
        ncols=2,
        kwargs=dict(sharex="all", sharey="all"),
        titles=["RS", "BO", "ASHA", "MOBSTER"],
        title_each_figure=True,
        legend_no=[0],
    )
    # The creation of ``results`` downloads files from S3 (only if
    # ``download_from_s3 == True``), reads the metadata and creates an inverse
    # index. If any result files are missing, or there are too many of them,
    # warning messages are printed
    results = ComparativeResults(
        experiment_names=experiment_names,
        setups=TMLR10_SETUPS,
        num_runs=num_runs,
        metadata_to_setup=metadata_to_setup,
        plot_params=plot_params,
        metadata_to_subplot=metadata_to_subplot,
        download_from_s3=download_from_s3,
    )
    # Create comparative plot (single panel)
    benchmark_name = "transformer_wikitext2"
    benchmark = benchmark_definitions(sagemaker_backend=True)[benchmark_name]
    # These parameters overwrite those given at construction
    plot_params = PlotParameters(
        metric=benchmark.metric,
        mode=benchmark.mode,
        ylim=(5, 8),
    )
    results.plot(
        benchmark_name=benchmark_name,
        plot_params=plot_params,
        file_name=f"./odsc-comparison-sagemaker-{benchmark_name}.png",
    )
