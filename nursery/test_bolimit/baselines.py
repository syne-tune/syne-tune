from syne_tune.experiments.default_baselines import (
    RandomSearch,
    BayesianOptimization,
)


class Methods:
    RS = "RS"
    BO_500_10 = "BO-500-10"
    BO_500_25 = "BO-500-25"
    BO_500_50 = "BO-500-50"
    BO_1000_10 = "BO-1000-10"
    BO_1000_25 = "BO-1000-25"
    BO_1000_50 = "BO-1000-50"


methods = {
    Methods.RS: lambda method_arguments: RandomSearch(method_arguments),
    Methods.BO_500_10: lambda method_arguments: BayesianOptimization(
        method_arguments,
        search_options=dict(
            max_size_data_for_model=500,
            max_size_top_fraction=0.1,
        ),
    ),
    Methods.BO_500_25: lambda method_arguments: BayesianOptimization(
        method_arguments,
        search_options=dict(
            max_size_data_for_model=500,
            max_size_top_fraction=0.25,
        ),
    ),
    Methods.BO_500_50: lambda method_arguments: BayesianOptimization(
        method_arguments,
        search_options=dict(
            max_size_data_for_model=500,
            max_size_top_fraction=0.5,
        ),
    ),
    Methods.BO_1000_10: lambda method_arguments: BayesianOptimization(
        method_arguments,
        search_options=dict(
            max_size_data_for_model=1000,
            max_size_top_fraction=0.1,
        ),
    ),
    Methods.BO_1000_25: lambda method_arguments: BayesianOptimization(
        method_arguments,
        search_options=dict(
            max_size_data_for_model=1000,
            max_size_top_fraction=0.25,
        ),
    ),
    Methods.BO_1000_50: lambda method_arguments: BayesianOptimization(
        method_arguments,
        search_options=dict(
            max_size_data_for_model=1000,
            max_size_top_fraction=0.5,
        ),
    ),
}
