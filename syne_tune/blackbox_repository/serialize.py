from pathlib import Path
from typing import Optional, Dict
import json
import syne_tune.config_space as sp


def serialize_configspace(
    path: str, configuration_space: Dict, fidelity_space: Optional[Dict] = None
):
    path = Path(path)
    with open(path / "configspace.json", "w") as f:
        json.dump({k: sp.to_dict(v) for k, v in configuration_space.items()}, f)

    if fidelity_space is not None:
        with open(path / "fidelityspace.json", "w") as f:
            json.dump({k: sp.to_dict(v) for k, v in fidelity_space.items()}, f)


def deserialize_configspace(path: str):
    def open_if_exists(name):
        config_path = Path(path) / name
        if config_path.exists():
            with open(config_path, "r") as file:
                cs_space = json.load(file)
                return {k: sp.from_dict(v) for k, v in cs_space.items()}
        else:
            return None

    configuration_space = open_if_exists("configspace.json")
    fidelity_space = open_if_exists("fidelityspace.json")
    return configuration_space, fidelity_space


def serialize_metadata(path: str, metadata):
    with open(path / "metadata.json", "w") as f:
        json.dump(metadata, f)


def deserialize_metadata(path: str):
    with open(Path(path) / "metadata.json", "r") as f:
        metadata = json.load(f)
        return metadata
