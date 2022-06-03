import logging
from pathlib import Path
from typing import List, Union, Dict, Optional

import s3fs as s3fs
from botocore.exceptions import NoCredentialsError

from syne_tune.blackbox_repository.blackbox import Blackbox
from syne_tune.blackbox_repository.blackbox_offline import (
    deserialize as deserialize_offline,
)
from syne_tune.blackbox_repository.blackbox_tabular import (
    deserialize as deserialize_tabular,
)

# where the blackbox repository is stored on s3
from syne_tune.blackbox_repository.conversion_scripts.recipes import (
    generate_blackbox_recipe,
)
from syne_tune.blackbox_repository.conversion_scripts.utils import (
    repository_path,
    s3_blackbox_folder,
)


def blackbox_list() -> List[str]:
    """
    :return: list of blackboxes available
    """
    return list(generate_blackbox_recipe.keys())


def load(
    name: str,
    skip_if_present: bool = True,
    s3_root: Optional[str] = None,
    generate_if_not_found: bool = True,
) -> Union[Dict[str, Blackbox], Blackbox]:
    """
    :param name: name of a blackbox present in the repository, see blackbox_list() to get list of available blackboxes
    :param skip_if_present: skip the download if the file locally exists
    :param s3_root: S3 root directory for blackbox repository. Defaults to
        S3 bucket name of SageMaker session
    :param generate_if_not_found: If the blackbox file is not present locally
        or on S3, should it be generated using its conversion script?
    :return: blackbox with the given name, download it if not present.
    """
    tgt_folder = Path(repository_path) / name
    if (
        tgt_folder.exists()
        and (tgt_folder / "metadata.json").exists()
        and skip_if_present
    ):
        logging.info(
            f"skipping download of {name} as {tgt_folder} already exists, change skip_if_present to redownload"
        )
    else:
        tgt_folder.mkdir(exist_ok=True, parents=True)
        try:
            s3_folder = s3_blackbox_folder(s3_root)
            fs = s3fs.S3FileSystem()
            data_on_s3 = fs.exists(f"{s3_folder}/{name}/metadata.json")
        except NoCredentialsError:
            data_on_s3 = False
        if data_on_s3:
            logging.info("found blackbox on S3, copying it locally")
            # download files from s3 to repository_path
            for src in fs.glob(f"{s3_folder}/{name}/*"):
                tgt = tgt_folder / Path(src).name
                logging.info(f"copying {src} to {tgt}")
                fs.get(src, str(tgt))
        else:
            assert generate_if_not_found, (
                "Blackbox files do not exist locally or on S3. If you have "
                + f"write permissions to {s3_folder}, you can set "
                + "generate_if_not_found=True in order to generate and persist them"
            )
            logging.info(
                "did not find blackbox files locally nor on S3, regenerating it locally and persisting it on S3."
            )
            generate_blackbox_recipe[name](s3_root=s3_root)

    if (tgt_folder / "hyperparameters.parquet").exists():
        return deserialize_tabular(tgt_folder)
    else:
        return deserialize_offline(tgt_folder)


if __name__ == "__main__":
    # list all blackboxes available
    blackboxes = blackbox_list()
    print(blackboxes)

    for bb in blackboxes:
        print(bb)
        # download an existing blackbox
        blackbox = load(bb)
        print(blackbox)
