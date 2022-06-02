from typing import Optional
import os
import logging
from functools import lru_cache
from pathlib import Path

import s3fs
import sagemaker
from botocore.exceptions import NoCredentialsError


@lru_cache(maxsize=1)
def s3_blackbox_folder(s3_root: Optional[str] = None):
    if s3_root is None:
        if "AWS_DEFAULT_REGION" not in os.environ:
            # avoids error "Must setup local AWS configuration with a region supported by SageMaker."
            # in case no region is explicitely configured
            os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
        s3_root = sagemaker.Session().default_bucket()

    return f"{s3_root}/blackbox-repository"


repository_path = Path("~/.blackbox-repository/").expanduser()


def upload(name: str, s3_root: Optional[str] = None):
    """
    Uploads a blackbox locally present in repository_path to S3.
    :param name: folder must be available in repository_path/name
    """
    try:
        fs = s3fs.S3FileSystem()
        for src in Path(repository_path / name).glob("*"):
            tgt = f"s3://{s3_blackbox_folder(s3_root)}/{name}/{src.name}"
            logging.info(f"copy {src} to {tgt}")
            fs.put(str(src), tgt)
    except NoCredentialsError:
        logging.warning(
            "Unable to locate credentials. Blackbox won't be uploaded to S3."
        )
