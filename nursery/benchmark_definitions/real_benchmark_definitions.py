from syne_tune.experiments.benchmark_definitions.common import RealBenchmarkDefinition
from benchmarking.benchmark_definitions.distilbert_on_imdb import (
    distilbert_imdb_benchmark,
)
from benchmarking.benchmark_definitions.finetune_transformer_glue import (
    finetune_transformer_glue_all_benchmarks,
)
from benchmarking.benchmark_definitions.finetune_transformer_swag import (
    finetune_transformer_swag_benchmark,
)
from benchmarking.benchmark_definitions.mlp_on_fashionmnist import (
    mlp_fashionmnist_benchmark,
)
from benchmarking.benchmark_definitions.lstm_wikitext2 import (
    lstm_wikitext2_benchmark,
)
from benchmarking.benchmark_definitions.resnet_cifar10 import (
    resnet_cifar10_benchmark,
)
from benchmarking.benchmark_definitions.transformer_wikitext2 import (
    transformer_wikitext2_benchmark,
)


def real_benchmark_definitions(
    sagemaker_backend: bool = False, **kwargs
) -> dict[str, RealBenchmarkDefinition]:
    result = {
        "resnet_cifar10": resnet_cifar10_benchmark(
            sagemaker_backend=sagemaker_backend, **kwargs
        ),
        "lstm_wikitext2": lstm_wikitext2_benchmark(
            sagemaker_backend=sagemaker_backend, **kwargs
        ),
        "mlp_fashionmnist": mlp_fashionmnist_benchmark(
            sagemaker_backend=sagemaker_backend, **kwargs
        ),
        "distilbert_imdb": distilbert_imdb_benchmark(
            sagemaker_backend=sagemaker_backend, **kwargs
        ),
        "transformer_wikitext2": transformer_wikitext2_benchmark(
            sagemaker_backend=sagemaker_backend, **kwargs
        ),
        "finetune_transformer_swag": finetune_transformer_swag_benchmark(
            sagemaker_backend=sagemaker_backend, **kwargs
        ),
    }
    result.update(
        finetune_transformer_glue_all_benchmarks(
            sagemaker_backend=sagemaker_backend, **kwargs
        )
    )
    return result
