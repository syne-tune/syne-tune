import copy

from syne_tune.experiments.default_baselines import (
    RandomSearch,
    BayesianOptimization,
    ASHA,
    MOBSTER,
)
from syne_tune.experiments.baselines import MethodArguments


class Methods:
    RS = "RS"
    BO = "BO"
    ASHA = "ASHA"
    MOBSTER = "MOBSTER"
    ASHA_TANH = "ASHA-TANH"
    MOBSTER_TANH = "MOBSTER-TANH"
    ASHA_RELU = "ASHA-RELU"
    MOBSTER_RELU = "MOBSTER-RELU"


def _modify_config_space(
    method_arguments: MethodArguments, value: str
) -> MethodArguments:
    result = copy.copy(method_arguments)
    result.config_space = dict(
        method_arguments.config_space,
        hp_activation_fn_1=value,
        hp_activation_fn_2=value,
    )
    return result


methods = {
    Methods.RS: lambda method_arguments: RandomSearch(method_arguments),
    Methods.BO: lambda method_arguments: BayesianOptimization(method_arguments),
    Methods.ASHA: lambda method_arguments: ASHA(
        method_arguments,
        type="promotion",
    ),
    Methods.MOBSTER: lambda method_arguments: MOBSTER(
        method_arguments,
        type="promotion",
    ),
    # Fix activations to "tanh"
    Methods.ASHA_TANH: lambda method_arguments: ASHA(
        _modify_config_space(method_arguments, value="tanh"),
        type="promotion",
    ),
    Methods.MOBSTER_TANH: lambda method_arguments: MOBSTER(
        _modify_config_space(method_arguments, value="tanh"),
        type="promotion",
    ),
    # Fix activations to "relu"
    Methods.ASHA_RELU: lambda method_arguments: ASHA(
        _modify_config_space(method_arguments, value="relu"),
        type="promotion",
    ),
    Methods.MOBSTER_RELU: lambda method_arguments: MOBSTER(
        _modify_config_space(method_arguments, value="relu"),
        type="promotion",
    ),
}
