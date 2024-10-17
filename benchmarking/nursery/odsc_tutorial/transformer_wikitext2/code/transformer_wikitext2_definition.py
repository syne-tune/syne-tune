from pathlib import Path

from transformer_wikitext2.code.training_script import (
    _config_space,
    METRIC_NAME,
    RESOURCE_ATTR,
    MAX_RESOURCE_ATTR,
)
from syne_tune.experiments.benchmark_definitions.common import RealBenchmarkDefinition
from syne_tune.remote.constants import (
    DEFAULT_GPU_INSTANCE_1GPU,
    DEFAULT_GPU_INSTANCE_4GPU,
)


def transformer_wikitext2_benchmark(sagemaker_backend: bool = False, **kwargs):
    if sagemaker_backend:
        instance_type = DEFAULT_GPU_INSTANCE_1GPU
    else:
        # For local backend, GPU cores serve different workers
        instance_type = DEFAULT_GPU_INSTANCE_4GPU
    fixed_parameters = dict(
        **{MAX_RESOURCE_ATTR: 40},
        d_model=256,
        ffn_ratio=1,
        nlayers=2,
        nhead=2,
        bptt=35,
        optimizer_name="sgd",
        input_data_dir="./",
        use_cuda=1,
        seed=1111,
        precision="float",
        log_interval=200,
    )
    config_space = {**_config_space, **fixed_parameters}
    _kwargs = dict(
        script=Path(__file__).parent / "training_script.py",
        config_space=config_space,
        metric=METRIC_NAME,
        mode="min",
        max_resource_attr=MAX_RESOURCE_ATTR,
        resource_attr=RESOURCE_ATTR,
        max_wallclock_time=5 * 3600,
        n_workers=4,
        instance_type=instance_type,
        framework="PyTorch",
    )
    _kwargs.update(kwargs)
    return RealBenchmarkDefinition(**_kwargs)
