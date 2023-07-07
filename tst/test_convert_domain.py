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

# This file has been taken from Ray. The reason for reusing the file is to be able to support the same API when
# defining search space while avoiding to have Ray as a required dependency. We may want to add functionality in the
# future.
import pytest

from syne_tune.utils import streamline_config_space
from syne_tune.config_space import (
    choice,
    finrange,
    logfinrange,
    logordinal,
    randint,
    lograndint,
    uniform,
    loguniform,
)


@pytest.mark.parametrize(
    "cs_original, cs_streamlined, exclude_names",
    [
        (
            {
                "hp_activation_fn_1": choice(["tanh", "relu"]),
                "hp_activation_fn_2": choice(["tanh", "relu"]),
                "hp_batch_size": choice([8, 16, 32, 64]),
                "hp_dropout_1": choice([0.0, 0.3, 0.6]),
                "hp_dropout_2": choice([0.0, 0.3, 0.6]),
                "hp_init_lr": choice([0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]),
                "hp_lr_schedule": choice(["cosine", "const"]),
                "hp_n_units_1": choice([16, 32, 64, 128, 256, 512]),
                "hp_n_units_2": choice([16, 32, 64, 128, 256, 512]),
            },
            {
                "hp_activation_fn_1": choice(["tanh", "relu"]),
                "hp_activation_fn_2": choice(["tanh", "relu"]),
                "hp_batch_size": logfinrange(8, 64, 4, cast_int=True),
                "hp_dropout_1": finrange(0.0, 0.6, 3),
                "hp_dropout_2": finrange(0.0, 0.6, 3),
                "hp_init_lr": logordinal([0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]),
                "hp_lr_schedule": choice(["cosine", "const"]),
                "hp_n_units_1": logfinrange(16, 512, 6, cast_int=True),
                "hp_n_units_2": logfinrange(16, 512, 6, cast_int=True),
            },
            None,
        ),
        (
            {"a": choice([1, 2, 5, 10, 20, 50])},
            {
                "a": logfinrange(
                    lower=0.9634924839989962,
                    upper=48.17462419994978,
                    size=6,
                    cast_int=True,
                )
            },
            None,
        ),
        (
            {
                "hp_activation_fn_1": choice(["tanh", "relu"]),
                "hp_activation_fn_2": choice(["tanh", "relu"]),
                "hp_batch_size": choice([8, 16, 32, 64]),
                "hp_dropout_1": choice([0.0, 0.3, 0.6]),
                "hp_dropout_2": choice([0.0, 0.3, 0.6]),
                "hp_init_lr": choice([0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]),
                "hp_lr_schedule": choice(["cosine", "const"]),
                "hp_n_units_1": choice([16, 32, 64, 128, 256, 512]),
                "hp_n_units_2": choice([16, 32, 64, 128, 256, 512]),
            },
            {
                "hp_activation_fn_1": choice(["tanh", "relu"]),
                "hp_activation_fn_2": choice(["tanh", "relu"]),
                "hp_batch_size": logfinrange(8, 64, 4, cast_int=True),
                "hp_dropout_1": finrange(0.0, 0.6, 3),
                "hp_dropout_2": choice([0.0, 0.3, 0.6]),
                "hp_init_lr": logordinal([0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]),
                "hp_lr_schedule": choice(["cosine", "const"]),
                "hp_n_units_1": choice([16, 32, 64, 128, 256, 512]),
                "hp_n_units_2": logfinrange(16, 512, 6, cast_int=True),
            },
            ["hp_dropout_2", "hp_n_units_1"],
        ),
        (
            {
                "a": choice([0.5, 2.0, 1.0, 4.0]),
                "b": randint(2, 500),
                "c": uniform(0.1, 0.9),
                "d": uniform(0.01, 10.0),
            },
            {
                "a": logfinrange(lower=0.5, upper=4.0, size=4),
                "b": lograndint(2, 500),
                "c": uniform(0.1, 0.9),
                "d": loguniform(0.01, 10.0),
            },
            None,
        ),
        (
            {
                "a": choice([0.1, 0.2, 0.3000001, 0.4, 0.5]),
                "b": choice(list(range(1000)) + [1001]),
            },
            {
                "a": finrange(lower=0.1, upper=0.5, size=5),
                "b": finrange(
                    lower=-0.4990009990009439,
                    upper=1000.500999000999,
                    size=1001,
                    cast_int=True,
                ),
            },
            None,
        ),
    ],
)
def test_streamline_config_space(cs_original, cs_streamlined, exclude_names):
    cs_out = streamline_config_space(
        config_space=cs_original, exclude_names=exclude_names, verbose=True
    )
    for name, domain in cs_streamlined.items():
        assert domain == cs_out[name], name
