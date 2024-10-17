from pathlib import Path
from typing import Optional, Dict
import json

from syne_tune.config_space import (
    config_space_from_json_dict,
    config_space_to_json_dict,
)
from syne_tune.util import dump_json_with_numpy


def serialize_configspace(
    path: str, configuration_space: Dict, fidelity_space: Optional[Dict] = None
):
    path = Path(path)
    dump_json_with_numpy(
        config_space_to_json_dict(configuration_space), path / "configspace.json"
    )
    if fidelity_space is not None:
        dump_json_with_numpy(
            config_space_to_json_dict(fidelity_space), path / "fidelityspace.json"
        )


def deserialize_configspace(path: str):
    def open_if_exists(name):
        config_path = Path(path) / name
        if config_path.exists():
            with open(config_path, "r") as file:
                return config_space_from_json_dict(json.load(file))
        else:
            return None

    configuration_space = open_if_exists("configspace.json")
    fidelity_space = open_if_exists("fidelityspace.json")
    return configuration_space, fidelity_space


def serialize_metadata(path: str, metadata: dict):
    dump_json_with_numpy(metadata, path / "metadata.json")


def deserialize_metadata(path: str):
    with open(Path(path) / "metadata.json", "r") as f:
        metadata = json.load(f)
        return metadata
