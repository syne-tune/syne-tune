from typing import Any, Optional
import logging

from transformer_wikitext2.baselines import methods
from transformer_wikitext2.benchmark_definitions import benchmark_definitions
from syne_tune.experiments import ComparativeResults, PlotParameters


SETUPS = list(methods.keys())


def metadata_to_setup(metadata: dict[str, Any]) -> Optional[str]:
    return metadata["algorithm"]


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    experiment_names = ("odsc-1",)
    num_runs = 10
    download_from_s3 = False  # Set ``True`` in order to download files from S3
    # Plot parameters across all benchmarks
    plot_params = PlotParameters(
        xlabel="wall-clock time",
        aggregate_mode="iqm_bootstrap",
        grid=True,
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
    # Create comparative plot (single panel)
    benchmark_name = "transformer_wikitext2"
    benchmark = benchmark_definitions(sagemaker_backend=False)[benchmark_name]
    # These parameters overwrite those given at construction
    plot_params = PlotParameters(
        metric=benchmark.metric,
        mode=benchmark.mode,
        ylim=(5, 8),
    )
    results.plot(
        benchmark_name=benchmark_name,
        plot_params=plot_params,
        file_name=f"./odsc-comparison-local-{benchmark_name}.png",
    )
