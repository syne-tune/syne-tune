from typing import Dict, Any, Optional
import logging

from transformer_wikitext2.baselines import methods
from transformer_wikitext2.benchmark_definitions import benchmark_definitions
from syne_tune.experiments import (
    TrialsOfExperimentResults,
    PlotParameters,
    MultiFidelityParameters,
    SubplotParameters,
)


SETUPS = list(methods.keys())


def metadata_to_setup(metadata: Dict[str, Any]) -> Optional[str]:
    return metadata["algorithm"]


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    experiment_names = ("odsc-1",)
    seed_to_plot = 0
    download_from_s3 = False  # Set ``True`` in order to download files from S3

    # Plot parameters across all benchmarks
    plot_params = PlotParameters(
        xlabel="wall-clock time",
        grid=True,
    )
    # We need to provide details about rung levels of the multi-fidelity methods.
    # Also, all methods compared are pause-and-resume
    multi_fidelity_params = MultiFidelityParameters(
        rung_levels=[1, 3, 9, 27, 40],
        multifidelity_setups={"ASHA": True, "MOBSTER": True},
    )
    # We would like to have 4 subfigures, one for each method
    plot_params.subplots = SubplotParameters(
        nrows=2,
        ncols=2,
        kwargs=dict(sharex="all", sharey="all"),
        titles=SETUPS,
        title_each_figure=True,
    )
    # The creation of ``results`` downloads files from S3 (only if
    # ``download_from_s3 == True``), reads the metadata and creates an inverse
    # index. If any result files are missing, or there are too many of them,
    # warning messages are printed
    results = TrialsOfExperimentResults(
        experiment_names=experiment_names,
        setups=SETUPS,
        metadata_to_setup=metadata_to_setup,
        plot_params=plot_params,
        multi_fidelity_params=multi_fidelity_params,
        download_from_s3=download_from_s3,
    )

    # Create plot for certain benchmark and seed
    benchmark_name = "transformer_wikitext2"
    benchmark = benchmark_definitions(sagemaker_backend=True)[benchmark_name]
    # These parameters overwrite those given at construction
    plot_params = PlotParameters(
        metric=benchmark.metric,
        mode=benchmark.mode,
    )
    results.plot(
        benchmark_name=benchmark_name,
        seed=seed_to_plot,
        plot_params=plot_params,
        file_name=f"./odsc-learncurves-local-seed{seed_to_plot}.png",
    )
