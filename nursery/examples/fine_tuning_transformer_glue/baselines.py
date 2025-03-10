from syne_tune.experiments.default_baselines import (
    BayesianOptimization,
    MOBSTER,
)


class Methods:
    BO = "BO"
    MOBSTER = "MOBSTER"


methods = {
    Methods.BO: lambda method_arguments: BayesianOptimization(method_arguments),
    Methods.MOBSTER: lambda method_arguments: MOBSTER(method_arguments),
}
