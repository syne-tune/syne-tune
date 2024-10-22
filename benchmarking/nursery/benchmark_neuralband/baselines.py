from typing import Dict, Any
from syne_tune.experiments.baselines import (
    search_options,
    default_arguments,
)
from syne_tune.experiments.default_baselines import (
    RandomSearch,
    BayesianOptimization,
    ASHA,
    MOBSTER,
    BOHB,
    KDE,
    convert_categorical_to_ordinal_numeric,
)
from syne_tune.optimizer.schedulers.neuralbands.neuralband import NeuralbandScheduler
from syne_tune.optimizer.schedulers.neuralbands.neuralband_supplement import (
    NeuralbandUCBScheduler,
    NeuralbandTSScheduler,
    NeuralbandEGreedyScheduler,
)


class Methods:
    RS = "RS"
    ASHA = "ASHA"
    GP = "GP"
    BOHB = "BOHB"
    MOBSTER = "MOB"
    TPE = "TPE"
    NeuralBandSH = "NeuralBandSH"
    NeuralBandHB = "NeuralBandHB"
    NeuralBand_UCB = "NeuralBandUCB"
    NeuralBand_TS = "NeuralBandTS"
    NeuralBandEpsilon = "NeuralBandEpsilon"


def conv_numeric_only(margs) -> Dict[str, Any]:
    return convert_categorical_to_ordinal_numeric(
        margs.config_space, kind=margs.fcnet_ordinal
    )


methods = {
    Methods.RS: lambda method_arguments: RandomSearch(method_arguments),
    Methods.ASHA: lambda method_arguments: ASHA(method_arguments),
    Methods.BOHB: lambda method_arguments: BOHB(method_arguments),
    Methods.TPE: lambda method_arguments: KDE(method_arguments),
    Methods.GP: lambda method_arguments: BayesianOptimization(method_arguments),
    Methods.MOBSTER: lambda method_arguments: MOBSTER(method_arguments),
    Methods.NeuralBandSH: lambda method_arguments: NeuralbandScheduler(
        **default_arguments(
            method_arguments,
            dict(
                config_space=conv_numeric_only(method_arguments),
                searcher="random",
                search_options=search_options(method_arguments),
                gamma=0.05,
                nu=0.02,
                max_while_loop=50,
                step_size=5,
                resource_attr=method_arguments.resource_attr,
            ),
        )
    ),
    Methods.NeuralBandHB: lambda method_arguments: NeuralbandScheduler(
        **default_arguments(
            method_arguments,
            dict(
                config_space=conv_numeric_only(method_arguments),
                searcher="random",
                search_options=search_options(method_arguments),
                gamma=0.04,
                nu=0.02,
                max_while_loop=50,
                step_size=5,
                resource_attr=method_arguments.resource_attr,
            ),
        )
    ),
    Methods.NeuralBand_UCB: lambda method_arguments: NeuralbandUCBScheduler(
        **default_arguments(
            method_arguments,
            dict(
                config_space=conv_numeric_only(method_arguments),
                searcher="random",
                search_options=search_options(method_arguments),
                lamdba=0.1,
                nu=0.001,
                max_while_loop=50,
                step_size=5,
                resource_attr=method_arguments.resource_attr,
            ),
        )
    ),
    Methods.NeuralBand_TS: lambda method_arguments: NeuralbandTSScheduler(
        **default_arguments(
            method_arguments,
            dict(
                config_space=conv_numeric_only(method_arguments),
                searcher="random",
                search_options=search_options(method_arguments),
                lamdba=0.1,
                nu=0.001,
                max_while_loop=50,
                step_size=5,
                resource_attr=method_arguments.resource_attr,
            ),
        )
    ),
    Methods.NeuralBandEpsilon: lambda method_arguments: NeuralbandEGreedyScheduler(
        **default_arguments(
            method_arguments,
            dict(
                config_space=conv_numeric_only(method_arguments),
                searcher="random",
                search_options=search_options(method_arguments),
                epsilon=0.1,
                max_while_loop=1000,
                step_size=5,
                resource_attr=method_arguments.resource_attr,
            ),
        )
    ),
}
