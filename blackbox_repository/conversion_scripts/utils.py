from functools import lru_cache
from pathlib import Path

import s3fs
import sagemaker


@lru_cache(maxsize=1)
def s3_blackbox_folder():
    return f"{sagemaker.Session().default_bucket()}/blackbox-repository"


repository_path = Path("~/.blackbox-repository/").expanduser()


def upload(name: str):
    """
    Uploads a blackbox locally present in repository_path to S3.
    :param name: folder must be available in repository_path/name
    """
    fs = s3fs.S3FileSystem()
    for src in Path(repository_path / name).glob("*"):
        tgt = f"s3://{s3_blackbox_folder()}/{name}/{src.name}"
        print(f"copy {src} to {tgt}")
        fs.put(str(src), tgt)
