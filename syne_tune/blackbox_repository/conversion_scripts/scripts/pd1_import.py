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
import logging
import tarfile

from syne_tune.blackbox_repository.conversion_scripts.blackbox_recipe import (
    BlackboxRecipe,
)
from syne_tune.blackbox_repository.conversion_scripts.utils import (
    repository_path,
    download_file,
)
from syne_tune.config_space import loguniform, uniform
from syne_tune.util import catchtime

logger = logging.getLogger(__name__)

CONFIGURATION_SPACE = {
    "lr_initial_value": loguniform(1e-5, 10),
    "lr_power": uniform(0.1, 2.0),
    "lr_decay_steps_factor": uniform(0.01, 0.99),
    "one_minus_momentum": loguniform(1e-3, 1.0),
}

DATA_TRANSFORM = {
    "hps.lr_hparams.initial_value": {
        "key": "lr_initial_value",
    },
    "hps.lr_hparams.power": {
        "key": "lr_power",
    },
    "hps.lr_hparams.decay_steps_factor": {
        "key": "lr_decay_steps_factor",
    },
    "hps.opt_hparams.momentum": {
        "key": "one_minus_momentum",
        "transform": lambda x: 1 - x,
    },
}

METRIC_TIME_THIS_RESOURCE = "metric_runtime"

RESOURCE_ATTR = "hp_epoch"


class PD1Recipe(BlackboxRecipe):
    def __init__(self):
        super(PD1Recipe, self).__init__(
            name="pd1",
            cite_reference="Pre-trained Gaussian processes for Bayesian optimization. "
            "Wang, Z. and Dahl G. and Swersky K. and Lee C. and Mariet Z. and Nado Z. and Gilmer J. and Snoek J. and Ghahramani Z. 2021.",
        )

    def _generate_on_disk(self):
        file_name = repository_path / "pd1.tar.gz"
        if not file_name.exists():
            logger.info(f"Did not find {file_name}. Starting download.")
            download_file(
                "http://storage.googleapis.com/gresearch/pint/pd1.tar.gz", file_name
            )
            with tarfile.open(file_name) as f:
                f.extractall(path=repository_path)
        else:
            logger.info(f"Skip downloading since {file_name} is available locally.")
