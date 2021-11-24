from blackbox_repository.conversion_scripts.scripts.nas201_import import generate_nas201
from blackbox_repository.conversion_scripts.scripts.fcnet_import import generate_fcnet
from blackbox_repository.conversion_scripts.scripts.icml2020_import import generate_deepar, generate_xgboost

generate_blackbox_recipe = {
    "icml-deepar": generate_deepar,
    "icml-xgboost": generate_xgboost,
    "nas201": generate_nas201,
    "fcnet": generate_fcnet,
}
