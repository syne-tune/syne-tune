from syne_tune.experiments.default_baselines import (
    BayesianOptimization,
    DyHPO,
    ASHA,
    MOBSTER,
    HyperTune,
)


class Methods:
    BO = "BO"
    ASHA = "ASHA"
    MOBSTER = "MOBSTER"
    HYPERTUNE = "HyperTune"
    DYHPO = "DYHPO"


methods = {
    Methods.BO: lambda method_arguments: BayesianOptimization(method_arguments),
    Methods.ASHA: lambda method_arguments: ASHA(method_arguments, type="promotion"),
    Methods.MOBSTER: lambda method_arguments: MOBSTER(
        method_arguments, type="promotion"
    ),
    Methods.HYPERTUNE: lambda method_arguments: HyperTune(
        method_arguments, type="promotion"
    ),
    Methods.DYHPO: lambda method_arguments: DyHPO(method_arguments),
}
