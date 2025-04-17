from benchmarking.benchmark_definitions import (
    SurrogateBenchmarkDefinition,
    fcnet_benchmark_definitions,
    lcbench_selected_benchmark_definitions,
)


def nas201_benchmark(dataset_name):
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=6 * 3600,
        n_workers=4,
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_error",
        mode="min",
        blackbox_name="nasbench201",
        dataset_name=dataset_name,
        max_resource_attr="epochs",
    )


nas201_benchmark_definitions = {
    "nas201-cifar10": nas201_benchmark("cifar10"),
    "nas201-cifar100": nas201_benchmark("cifar100"),
    "nas201-ImageNet16-120": nas201_benchmark("ImageNet16-120"),
}


benchmark_definitions = {
    **fcnet_benchmark_definitions,
    **nas201_benchmark_definitions,
    **lcbench_selected_benchmark_definitions,
}
