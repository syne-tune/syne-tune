from typing import Dict, Any
import argparse
import json

from syne_tune.constants import ST_CONFIG_JSON_FNAME_ARG


def add_config_json_to_argparse(parser: argparse.ArgumentParser):
    """
    To be called for the argument parser in the endpoint script.

    :param parser: Parser to add extra arguments to
    """
    parser.add_argument(f"--{ST_CONFIG_JSON_FNAME_ARG}", type=str)


def load_config_json(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loads configuration from JSON file and returns the union with ``args``.

    :param args: Arguments returned by ``ArgumentParser``, as dictionary
    :return: Combined configuration dictionary
    """
    with open(args[ST_CONFIG_JSON_FNAME_ARG], "r") as f:
        config = json.load(f)
    config.update(args)
    return config
