# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
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


# add a blackbox recipe here to expose it in Syne Tune
from syne_tune.blackbox_repository.conversion_scripts.scripts.pd1_import import (
    PD1Recipe,
)

recipes = [
    DeepARRecipe(),
    XGBoostRecipe(),
    NASBench201Recipe(),
    FCNETRecipe(),
    LCBenchRecipe(),
    PD1Recipe(),
]


from syne_tune.try_import import try_import_yahpo_message

try:
    from syne_tune.blackbox_repository.conversion_scripts.scripts.yahpo_import import (
        YAHPORecipe,
        yahpo_scenarios,
    )

    for scenario in yahpo_scenarios:
        recipes.append(YAHPORecipe("yahpo-" + scenario))
except ImportError:
    print(try_import_yahpo_message())


generate_blackbox_recipes = {recipe.name: recipe for recipe in recipes}
