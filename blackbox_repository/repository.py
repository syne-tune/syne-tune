from pathlib import Path
from typing import List

import s3fs as s3fs
import sagemaker

from blackbox_repository import BlackboxOffline
from blackbox_repository.blackbox_offline import deserialize as deserialize_offline
from blackbox_repository.blackbox_tabular import deserialize as deserialize_tabular

# where the blackbox repository is stored on s3
from blackbox_repository.conversion_scripts.recipes import generate_blackbox_recipe
from blackbox_repository.conversion_scripts.utils import repository_path, s3_blackbox_folder


def blackbox_list() -> List[str]:
    """
    :return: list of blackboxes available
    """
    return list(generate_blackbox_recipe.keys())


def load(name: str, skip_if_present: bool = True) -> BlackboxOffline:
    """
    :param name: name of a blackbox present in the repository, see list() to get list of available blackboxes
    :param skip_if_present: skip the download if the file locally exists
    :return: blackbox with the given name, download it if not present.
    """
    tgt_folder = Path(repository_path) / name
    if tgt_folder.exists() and (tgt_folder / "metadata.json").exists() and skip_if_present:
        print(f"skipping download of {name} as {tgt_folder} already exists, change skip_if_present to redownload")
    else:
        tgt_folder.mkdir(exist_ok=True, parents=True)
        fs = s3fs.S3FileSystem()
        data_on_s3 = fs.exists(f"{s3_blackbox_folder}/{name}/metadata.json")
        if data_on_s3:
            # download files from s3 to repository_path
            for src in fs.glob(f"{s3_blackbox_folder}/{name}/*"):
                tgt = tgt_folder / Path(src).name
                print(f"copying {src} to {tgt}")
                fs.get(src, str(tgt))
        else:
            print("did not find blackbox files locally nor on s3, regenerating it locally and persisting it on S3.")
            generate_blackbox_recipe[name]()

    if (tgt_folder / "hyperparameters.parquet").exists():
        return deserialize_tabular(tgt_folder)
    else:
        return deserialize_offline(tgt_folder)


if __name__ == '__main__':
    # list all blackboxes available
    blackboxes = blackbox_list()
    print(blackboxes)

    for bb in blackboxes:
        print(bb)
        # download an existing blackbox
        blackbox = load(bb)
        print(blackbox)