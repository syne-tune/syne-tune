from syne_tune.experiments.default_baselines import (
    RandomSearch,
    BayesianOptimization,
    ASHA,
    MOBSTER,
)


class Methods:
    RS = "RS"
    BO = "BO"
    ASHA = "ASHA"
    MOBSTER = "MOBSTER"


methods = {
    Methods.RS: lambda method_arguments: RandomSearch(method_arguments),
    Methods.BO: lambda method_arguments: BayesianOptimization(method_arguments),
    Methods.ASHA: lambda method_arguments: ASHA(method_arguments),
    Methods.MOBSTER: lambda method_arguments: MOBSTER(method_arguments),
}
