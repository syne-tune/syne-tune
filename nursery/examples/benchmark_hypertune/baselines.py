from syne_tune.experiments.default_baselines import (
    ASHA,
    MOBSTER,
    HyperTune,
    SyncHyperband,
    SyncBOHB,
)


class Methods:
    ASHA = "ASHA"
    MOBSTER_JOINT = "MOBSTER-JOINT"
    MOBSTER_INDEP = "MOBSTER-INDEP"
    HYPERTUNE_INDEP = "HYPERTUNE-INDEP"
    HYPERTUNE_JOINT = "HYPERTUNE-JOINT"
    SYNCHB = "SYNCHB"
    BOHB = "BOHB"


methods = {
    Methods.ASHA: lambda method_arguments: ASHA(
        method_arguments,
        type="promotion",
    ),
    Methods.MOBSTER_JOINT: lambda method_arguments: MOBSTER(
        method_arguments,
        type="promotion",
    ),
    Methods.MOBSTER_INDEP: lambda method_arguments: MOBSTER(
        method_arguments,
        type="promotion",
        search_options=dict(model="gp_independent"),
    ),
    Methods.HYPERTUNE_INDEP: lambda method_arguments: HyperTune(
        method_arguments,
        type="promotion",
        search_options=dict(model="gp_independent"),
    ),
    Methods.HYPERTUNE_JOINT: lambda method_arguments: HyperTune(
        method_arguments,
        type="promotion",
        search_options=dict(model="gp_multitask"),
    ),
    Methods.SYNCHB: lambda method_arguments: SyncHyperband(method_arguments),
    Methods.BOHB: lambda method_arguments: SyncBOHB(method_arguments),
}
