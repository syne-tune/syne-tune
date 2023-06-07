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

from syne_tune.experiments import (
    ComparativeResults,
    PlotParameters,
    ShowTrialParameters,
)
from benchmarking.examples.fine_tuning_transformer_glue.baselines import methods
from benchmarking.commons.benchmark_definitions.real_benchmark_definitions import (
    real_benchmark_definitions,
)


SETUPS = list(methods.keys())


def metadata_to_setup(metadata: Dict[str, Any]) -> Optional[str]:
    # The setup is the algorithm. No filtering
    return metadata["algorithm"]


if __name__ == "__main__":
    experiment_name = "glue-6"
    experiment_names = (experiment_name,)
    num_runs = 5
    download_from_s3 = False  # Set ``True`` in order to download files from S3
    # Plot parameters across all benchmarks
    plot_params = PlotParameters(
        xlabel="wall-clock time",
        aggregate_mode="iqm_bootstrap",
        grid=True,
    )
    # We also show the performance of the initial trial, which corresponds to the
    # Hugging Face default
    plot_params.show_init_trials = ShowTrialParameters(
        setup_name="BO",
        trial_id=0,
        new_setup_name="default",
    )
    # The creation of ``results`` downloads files from S3 (only if
    # ``download_from_s3 == True``), reads the metadata and creates an inverse
    # index. If any result files are missing, or there are too many of them,
    # warning messages are printed
    results = ComparativeResults(
        experiment_names=experiment_names,
        setups=SETUPS,
        num_runs=num_runs,
        metadata_to_setup=metadata_to_setup,
        plot_params=plot_params,
        download_from_s3=download_from_s3,
    )

    # We can now create plots for the different benchmarks
    for dataset, ylim in [
        ("rte", (0.27, 0.38)),
        ("mrpc", (0.09, 0.15)),
        ("stsb", (0.1, 0.15)),
    ]:
        for do_modsel in [False, True]:
            if do_modsel:
                benchmark_name = f"finetune_transformer_glue_modsel_{dataset}"
                title = f"Fine-tuning and model selection on GLUE {dataset}"
            else:
                benchmark_name = f"finetune_transformer_glue_{dataset}"
                title = f"Fine-tuning bert-base-cased on GLUE {dataset}"
            benchmark = real_benchmark_definitions()[benchmark_name]
            # These parameters overwrite those given at construction
            plot_params = PlotParameters(
                title=title,
                metric=benchmark.metric,
                mode=benchmark.mode,
                ylim=ylim,
            )
            results.plot(
                benchmark_name=benchmark_name,
                plot_params=plot_params,
                file_name=f"./{benchmark_name}.png",
            )
