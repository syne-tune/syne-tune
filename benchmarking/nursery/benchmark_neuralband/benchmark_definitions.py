from syne_tune.experiments.benchmark_definitions import (
    nas201_benchmark_definitions,
    fcnet_benchmark_definitions,
    lcbench_selected_benchmark_definitions,
)


benchmark_definitions = {
    **nas201_benchmark_definitions,
    **fcnet_benchmark_definitions,
    **lcbench_selected_benchmark_definitions,
}
