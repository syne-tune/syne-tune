from pathlib import Path

from syne_tune.experiments.benchmark_definitions.common import RealBenchmarkDefinition
from benchmarking.training_scripts.mlp_on_fashion_mnist.mlp_on_fashion_mnist import (
    _config_space,
    METRIC_NAME,
    RESOURCE_ATTR,
)
from syne_tune.remote.constants import DEFAULT_CPU_INSTANCE


def mlp_fashionmnist_benchmark(sagemaker_backend: bool = False, **kwargs):
    config_space = dict(
        _config_space,
        dataset_path="./",
        epochs=81,
        report_current_best="False",
    )
    _kwargs = dict(
        script=Path(__file__).parent.parent
        / "training_scripts"
        / "mlp_on_fashion_mnist"
        / "mlp_on_fashion_mnist.py",
        config_space=config_space,
        max_wallclock_time=3 * 3600,  # TODO
        n_workers=4,
        instance_type=DEFAULT_CPU_INSTANCE,
        metric=METRIC_NAME,
        mode="max",
        max_resource_attr="epochs",
        resource_attr=RESOURCE_ATTR,
        framework="PyTorch",
    )
    _kwargs.update(kwargs)
    return RealBenchmarkDefinition(**_kwargs)


# Support for cost models
#
# from benchmarking.training_scripts.mlp_on_fashion_mnist.mlp_on_fashion_mnist import (
#     NUM_UNITS_1,
#     NUM_UNITS_2,
# )
#
# def get_cost_model(params):
#     """
#     This cost model ignores the batch size, but depends on the number of units
#     in the two layers only.
#     """
#     try:
#         from syne_tune.optimizer.schedulers.searchers.bayesopt.models.cost.linear_cost_model import (
#             FixedLayersMLPCostModel,
#         )
#
#         num_inputs = 28 * 28
#         num_outputs = 10
#         num_units_keys = [NUM_UNITS_1, NUM_UNITS_2]
#         (
#             expected_hidden_layer_width,
#             exp_vals,
#         ) = FixedLayersMLPCostModel.get_expected_hidden_layer_width(
#             _config_space, num_units_keys
#         )
#         return FixedLayersMLPCostModel(
#             num_inputs=num_inputs,
#             num_outputs=num_outputs,
#             num_units_keys=num_units_keys,
#             expected_hidden_layer_width=expected_hidden_layer_width,
#         )
#     except Exception:
#         return None
