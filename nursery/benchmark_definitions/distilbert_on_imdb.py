from pathlib import Path

from benchmarking.training_scripts.distilbert_on_imdb.distilbert_on_imdb import (
    METRIC_ACCURACY,
    RESOURCE_ATTR,
    _config_space,
)
from syne_tune.experiments.benchmark_definitions.common import RealBenchmarkDefinition
from syne_tune.remote.constants import DEFAULT_GPU_INSTANCE_1GPU


def distilbert_imdb_benchmark(sagemaker_backend: bool = False, **kwargs):
    config_space = dict(
        _config_space,
        dataset_path="./",
        epochs=15,
    )
    _kwargs = dict(
        script=Path(__file__).parent.parent
        / "training_scripts"
        / "distilbert_on_imdb"
        / "distilbert_on_imdb.py",
        config_space=config_space,
        max_wallclock_time=3 * 3600,  # TODO
        n_workers=4,
        instance_type=DEFAULT_GPU_INSTANCE_1GPU,
        metric=METRIC_ACCURACY,
        mode="max",
        max_resource_attr="epochs",
        resource_attr=RESOURCE_ATTR,
        framework="HuggingFace",
    )
    _kwargs.update(kwargs)
    return RealBenchmarkDefinition(**_kwargs)
