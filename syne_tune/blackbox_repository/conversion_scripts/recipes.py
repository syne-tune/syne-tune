import logging

from syne_tune.blackbox_repository.conversion_scripts.scripts.icml2020_import import (
    DeepARRecipe,
    XGBoostRecipe,
)
from syne_tune.blackbox_repository.conversion_scripts.scripts.lcbench.lcbench import (
    LCBenchRecipe,
)
from syne_tune.blackbox_repository.conversion_scripts.scripts.nasbench201_import import (
    NASBench201Recipe,
)
from syne_tune.blackbox_repository.conversion_scripts.scripts.fcnet_import import (
    FCNETRecipe,
)
from syne_tune.blackbox_repository.conversion_scripts.scripts.pd1_import import (
    PD1Recipe,
)


# add a blackbox recipe here to expose it in Syne Tune
recipes = [
    DeepARRecipe(),
    XGBoostRecipe(),
    NASBench201Recipe(),
    FCNETRecipe(),
    LCBenchRecipe(),
    PD1Recipe(),
]


try:
    from syne_tune.blackbox_repository.conversion_scripts.scripts.yahpo_import import (
        YAHPORecipe,
        yahpo_scenarios,
    )

    for scenario in yahpo_scenarios:
        recipes.append(YAHPORecipe("yahpo-" + scenario))
except ImportError as e:
    logging.debug(e)


generate_blackbox_recipes = {recipe.name: recipe for recipe in recipes}
