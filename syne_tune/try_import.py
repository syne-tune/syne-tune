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
from typing import Optional


def try_import_gpsearchers_message() -> str:
    return _try_import_message(
        "Gaussian process based searchers are not imported", tag="gpsearchers"
    )


def try_import_kde_message() -> str:
    return _try_import_message("KDE searchers are not imported", tag="kde")


def try_import_bore_message() -> str:
    return _try_import_message(
        "BORE searchers are not imported (not contained in extra)", tag="bore"
    )


def try_import_raytune_message() -> str:
    return _try_import_message(
        "Ray Tune schedulers and searchers are not imported", tag="raytune"
    )


def try_import_benchmarks_message() -> str:
    return _try_import_message(
        "Dependencies for benchmarks are not imported", tag="benchmarks"
    )


def try_import_aws_message() -> str:
    return _try_import_message("AWS dependencies are not imported", tag="aws")


def try_import_botorch_message() -> str:
    return _try_import_message(
        "BoTorch dependencies are not imported (needs Python 3.8 or later)",
        tag="botorch",
    )


def try_import_blackbox_repository_message() -> str:
    return _try_import_message(
        "Dependencies of blackbox repository are not imported",
        tag="blackbox-repository",
    )


def try_import_yahpo_message() -> str:
    return _try_import_message(
        "Dependencies of YAHPO are not imported",
        tag="yahpo",
    )


def try_import_moo_message() -> str:
    return _try_import_message(
        "Multi Objective Optimization dependencies are not imported", tag="moo"
    )


def try_import_visual_message() -> str:
    return _try_import_message(
        "Dependencies for visualization are not imported", tag="visual"
    )


def try_import_backends_message() -> str:
    return _try_import_message(
        "LocalBackend / PythonBackend are not imported", tag=None
    )


def _try_import_message(message_text: str, tag: Optional[str]) -> str:
    if tag is None:
        insert = ""
    else:
        insert = "[" + tag + "]"
    return (
        message_text
        + " since dependencies are missing. You can install them with\n"
        + f"   pip install 'syne-tune{insert}'\n"
        + "or (for everything)\n"
        + "   pip install 'syne-tune[extra]'"
    )
