from syne_tune.experiments.default_baselines import (
    RandomSearch,
    BayesianOptimization,
    ASHA,
    MOBSTER,
)


class Methods:
    RS = "RS"
    ASHA = "ASHA"
    BO = "BO"
    BO_WARP = "BO-WARP"
    BO_BOXCOX = "BO-BOXCOX"
    BO_WARP_BOXCOX = "BO-WARP-BOXCOX"
    MOBSTER = "MOBSTER"
    MOBSTER_WARP = "MOBSTER-WARP"
    MOBSTER_BOXCOX = "MOBSTER-BOXCOX"
    MOBSTER_WARP_BOXCOX = "MOBSTER-WARP-BOXCOX"


methods = {
    Methods.RS: lambda method_arguments: RandomSearch(method_arguments),
    Methods.ASHA: lambda method_arguments: ASHA(
        method_arguments,
        type="promotion",
    ),
    Methods.BO: lambda method_arguments: BayesianOptimization(method_arguments),
    Methods.BO_WARP: lambda method_arguments: BayesianOptimization(
        method_arguments,
        search_options=dict(input_warping=True),
    ),
    Methods.BO_BOXCOX: lambda method_arguments: BayesianOptimization(
        method_arguments,
        search_options=dict(boxcox_transform=True),
    ),
    Methods.BO_WARP_BOXCOX: lambda method_arguments: BayesianOptimization(
        method_arguments,
        search_options=dict(input_warping=True, boxcox_transform=True),
    ),
    Methods.MOBSTER: lambda method_arguments: MOBSTER(
        method_arguments,
        type="promotion",
    ),
    Methods.MOBSTER_WARP: lambda method_arguments: MOBSTER(
        method_arguments,
        type="promotion",
        search_options=dict(input_warping=True),
    ),
    Methods.MOBSTER_BOXCOX: lambda method_arguments: MOBSTER(
        method_arguments,
        type="promotion",
        search_options=dict(boxcox_transform=True),
    ),
    Methods.MOBSTER_WARP_BOXCOX: lambda method_arguments: MOBSTER(
        method_arguments,
        type="promotion",
        search_options=dict(input_warping=True, boxcox_transform=True),
    ),
}
