from typing import Dict, Any
from syne_tune.experiments.baselines import (
    convert_categorical_to_ordinal,
    convert_categorical_to_ordinal_numeric,
)
from syne_tune.experiments.default_baselines import (
    ASHA,
    SyncHyperband,
    SyncBOHB,
    DEHB,
    SyncMOBSTER,
)


class Methods:
    ASHA = "ASHA"
    SYNCHB = "SYNCHB"
    DEHB = "DEHB"
    BOHB = "BOHB"
    ASHA_ORD = "ASHA-ORD"
    SYNCHB_ORD = "SYNCHB-ORD"
    DEHB_ORD = "DEHB-ORD"
    BOHB_ORD = "BOHB-ORD"
    ASHA_STOP = "ASHA-STOP"
    SYNCMOBSTER = "SYNCMOBSTER"


def conv_numeric_then_rest(margs) -> Dict[str, Any]:
    return convert_categorical_to_ordinal(
        convert_categorical_to_ordinal_numeric(
            margs.config_space, kind=margs.fcnet_ordinal
        )
    )


methods = {
    Methods.ASHA: lambda method_arguments: ASHA(
        method_arguments,
        type="promotion",
    ),
    Methods.SYNCHB: lambda method_arguments: SyncHyperband(method_arguments),
    Methods.DEHB: lambda method_arguments: DEHB(method_arguments),
    Methods.BOHB: lambda method_arguments: SyncBOHB(method_arguments),
    Methods.ASHA_ORD: lambda method_arguments: ASHA(
        method_arguments,
        config_space=conv_numeric_then_rest(method_arguments),
        type="promotion",
    ),
    Methods.SYNCHB_ORD: lambda method_arguments: SyncHyperband(
        method_arguments,
        config_space=conv_numeric_then_rest(method_arguments),
    ),
    Methods.DEHB_ORD: lambda method_arguments: DEHB(
        method_arguments,
        config_space=conv_numeric_then_rest(method_arguments),
    ),
    Methods.BOHB_ORD: lambda method_arguments: SyncBOHB(
        method_arguments,
        config_space=conv_numeric_then_rest(method_arguments),
    ),
    Methods.ASHA_STOP: lambda method_arguments: ASHA(
        method_arguments,
        type="stopping",
    ),
    Methods.SYNCMOBSTER: lambda method_arguments: SyncMOBSTER(method_arguments),
}
