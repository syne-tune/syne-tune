import os
from pathlib import Path
from typing import Optional

repository_path = Path("~/.blackbox-repository/").expanduser()
repo_id = "synetune/blackbox-repository"


def get_sub_directory_and_name(name: str):
    """
    Blackboxes are either stored under "{blackbox-repository}/{name}" (such as fcnet, nas201, ...) or
    "{blackbox-repository}/{subdir}/{subname}" for all yahpo benchmark. In the Yahpo case, "yahpo-rbv2_xgboost"
    is for instance stored on "{blackbox-repository}/yahpo/rbv2_xgboost/".
    :param name: name of the blackbox, for instance "fcnet", "lcbench" or "yahpo-rbv2_xgboost".
    :return: subdirectory and subname such that the blackbox should be stored on {blackbox_repository}/{subdir}/{name}.
    """
    if name.startswith("yahpo-"):
        return "yahpo", name[6:]
    else:
        return ".", name


def blackbox_local_path(name: str) -> Path:
    subdir, subname = get_sub_directory_and_name(name)
    return Path(repository_path) / subdir / subname


def upload_blackbox(name: str, custom_repo_id: Optional[str] = None):
    """
    Uploads a blackbox locally present in repository_path to HuggingFace hub
    :param name: folder must be available in repository_path/name
    :param custom_repo_id: hugging face hub where the blackbox should be addded
    """
    from huggingface_hub import HfApi

    if name.startswith("yahpo"):
        _, subname = get_sub_directory_and_name(name)
        path_in_repo = f"yahpo/{subname}"
    else:
        path_in_repo = name

    HfApi().upload_folder(
        folder_path=blackbox_local_path(name),
        path_in_repo=path_in_repo,
        repo_id=repo_id if not custom_repo_id else custom_repo_id,
        repo_type="dataset",
        commit_message=f"Upload blackbox {name}",
        token=os.getenv("HF_TOKEN"),
    )


def download_file(source: str, destination: str):
    import shutil
    import requests
    from syne_tune.util import catchtime

    with catchtime("Downloading file."):
        with requests.get(source, stream=True) as r:
            with open(destination, "wb") as f:
                shutil.copyfileobj(r.raw, f)
