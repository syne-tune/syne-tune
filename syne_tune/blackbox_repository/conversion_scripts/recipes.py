from syne_tune.blackbox_repository.conversion_scripts.scripts.lcbench.lcbench import (
    generate_lcbench,
)
from syne_tune.blackbox_repository.conversion_scripts.scripts.nasbench201_import import (
    generate_nasbench201,
)
from syne_tune.blackbox_repository.conversion_scripts.scripts.fcnet_import import (
    generate_fcnet,
)
from syne_tune.blackbox_repository.conversion_scripts.scripts.icml2020_import import (
    generate_deepar,
    generate_xgboost,
)

generate_blackbox_recipe = {
    "icml-deepar": generate_deepar,
    "icml-xgboost": generate_xgboost,
    "nasbench201": generate_nasbench201,
    "fcnet": generate_fcnet,
    "lcbench": generate_lcbench,
}
