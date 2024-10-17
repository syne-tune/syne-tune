from syne_tune.experiments.benchmark_definitions import (
    yahpo_iaml_benchmark_definitions,
    yahpo_rbv2_benchmark_definitions,
    yahpo_nb301_benchmark_definitions,
)
from syne_tune.experiments.benchmark_definitions.yahpo import (
    yahpo_rbv2_metrics,
    yahpo_iaml_methods,
    yahpo_rbv2_methods,
)


# RESTRICT_FIDELITIES = False
RESTRICT_FIDELITIES = True


benchmark_definitions_iaml = {
    k: v
    for method in yahpo_iaml_methods
    for k, v in yahpo_iaml_benchmark_definitions(
        method, restrict_fidelities=RESTRICT_FIDELITIES
    ).items()
}


benchmark_definitions_rbv2 = dict()
for method in yahpo_rbv2_methods:
    definition = yahpo_rbv2_benchmark_definitions(
        method, restrict_fidelities=RESTRICT_FIDELITIES
    )
    for metric in yahpo_rbv2_metrics:
        key = f"rbv2-{method}-{metric[0]}"
        prefix = f"yahpo-rbv2_{method}_{metric[0]}"
        benchmark_definitions_rbv2[key] = {
            k: v for k, v in definition.items() if k.startswith(prefix)
        }


# benchmark_definitions = benchmark_definitions_iaml
# benchmark_definitions = benchmark_definitions_rbv2
benchmark_definitions = yahpo_nb301_benchmark_definitions
