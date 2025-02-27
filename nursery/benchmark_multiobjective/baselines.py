from syne_tune.experiments.default_baselines import (
    RandomSearch,
    MOREA,
    LSOBO,
)


class Methods:
    RS = "RS"
    MOREA = "MOREA"
    LSOBO = "LSOBO"


methods = {
    Methods.RS: lambda method_arguments: RandomSearch(method_arguments),
    Methods.MOREA: lambda method_arguments: MOREA(
        method_arguments, population_size=10, sample_size=5
    ),
    Methods.LSOBO: lambda method_arguments: LSOBO(
        method_arguments, searcher="bayesopt"
    ),
}
