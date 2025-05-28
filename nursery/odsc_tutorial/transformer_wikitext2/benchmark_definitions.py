from syne_tune.experiments.benchmark_definitions import RealBenchmarkDefinition
from transformer_wikitext2.code.transformer_wikitext2_definition import (
    transformer_wikitext2_benchmark,
)


def benchmark_definitions(
    sagemaker_backend: bool = False, **kwargs
) -> dict[str, RealBenchmarkDefinition]:
    return {
        "transformer_wikitext2": transformer_wikitext2_benchmark(
            sagemaker_backend=sagemaker_backend, **kwargs
        ),
    }
