from typing import Dict, Any, Optional
import logging

from baselines import methods
from benchmark_definitions import benchmark_definitions
from syne_tune.experiments import ComparativeResults, PlotParameters, SubplotParameters


def metadata_to_setup(metadata: Dict[str, Any]) -> Optional[str]:
    # The setup is the algorithm. No filtering
    return metadata["algorithm"]


SETUPS_RIGHT = ("ASHA", "SYNCHB", "BOHB")


def metadata_to_subplot(metadata: Dict[str, Any]) -> Optional[int]:
    return int(metadata["algorithm"] in SETUPS_RIGHT)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    experiment_name = "docs-1"
    experiment_names = (experiment_name,)
    setups = list(methods.keys())
    num_runs = 15
    download_from_s3 = False  # Set ``True`` in order to download files from S3
    # Plot parameters across all benchmarks
    plot_params = PlotParameters(
        xlabel="wall-clock time",
        aggregate_mode="iqm_bootstrap",
        grid=True,
    )
    # We would like two subplots (1 row, 2 columns), with MOBSTER and HYPERTUNE
    # results on the left, and the remaining baselines on the right. Each
    # column gets its own title, and legends are shown in both
    plot_params.subplots = SubplotParameters(
        nrows=1,
        ncols=2,
        kwargs=dict(sharey="all"),
        titles=["Model-based Methods", "Baselines"],
        legend_no=[0, 1],
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
    # We can now create plots for the different benchmarks
    # First: nas201-cifar100
    benchmark_name = "nas201-cifar100"
    benchmark = benchmark_definitions[benchmark_name]
    # These parameters overwrite those given at construction
    plot_params = PlotParameters(
        metric=benchmark.metric,
        mode=benchmark.mode,
        ylim=(0.265, 0.31),
    )
    results.plot(
        benchmark_name=benchmark_name,
        plot_params=plot_params,
        file_name=f"./{experiment_name}-{benchmark_name}.png",
    )
    # Next: nas201-ImageNet16-120
    benchmark_name = "nas201-ImageNet16-120"
    benchmark = benchmark_definitions[benchmark_name]
    # These parameters overwrite those given at construction
    plot_params = PlotParameters(
        metric=benchmark.metric,
        mode=benchmark.mode,
        ylim=(0.535, 0.58),
    )
    results.plot(
        benchmark_name=benchmark_name,
        plot_params=plot_params,
        file_name=f"./{experiment_name}-{benchmark_name}.png",
    )
