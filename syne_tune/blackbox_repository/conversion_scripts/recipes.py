import logging

from syne_tune.blackbox_repository.conversion_scripts.scripts.hpob_import import (
    HPOBRecipe4796,
    HPOBRecipe5527,
    HPOBRecipe5636,
    HPOBRecipe5859,
    HPOBRecipe5860,
    HPOBRecipe5891,
    HPOBRecipe5906,
    HPOBRecipe5965,
    HPOBRecipe5970,
    HPOBRecipe5971,
    HPOBRecipe6766,
    HPOBRecipe6767,
    HPOBRecipe6794,
    HPOBRecipe7607,
    HPOBRecipe7609,
    HPOBRecipe5889,
)
from syne_tune.blackbox_repository.conversion_scripts.scripts.tabrepo_import import (
    TabrepoRandomForest,
    TabrepoLinearModel,
    TabrepoCatBoost,
    TabrepoXGBoost,
    TabrepoExtraTrees,
    TabrepoNeuralNetTorch,
    TabrepoLightGBM,
    TabrepoRecipeKNeighbors,
)
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
    HPOBRecipe4796(),
    HPOBRecipe5527(),
    HPOBRecipe5636(),
    HPOBRecipe5859(),
    HPOBRecipe5860(),
    HPOBRecipe5891(),
    HPOBRecipe5906(),
    HPOBRecipe5965(),
    HPOBRecipe5970(),
    HPOBRecipe5971(),
    HPOBRecipe6766(),
    HPOBRecipe6767(),
    HPOBRecipe6794(),
    HPOBRecipe7607(),
    HPOBRecipe7609(),
    HPOBRecipe5889(),
    TabrepoRandomForest(),
    TabrepoLinearModel(),
    TabrepoCatBoost(),
    TabrepoXGBoost(),
    TabrepoExtraTrees(),
    TabrepoNeuralNetTorch(),
    TabrepoLightGBM(),
    TabrepoRecipeKNeighbors(),
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
