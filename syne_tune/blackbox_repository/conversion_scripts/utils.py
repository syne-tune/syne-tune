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
import os
import logging
import hashlib
from functools import lru_cache
from pathlib import Path

from syne_tune.try_import import try_import_aws_message

try:
    import s3fs
    import sagemaker
    from botocore.exceptions import NoCredentialsError
except ImportError:
    print(try_import_aws_message())


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


def upload_blackbox(name: str, s3_root: Optional[str] = None):
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


def download_file(source: str, destination: str):
    import shutil
    import requests
    from syne_tune.util import catchtime

    with catchtime("Downloading file."):
        with requests.get(source, stream=True) as r:
            with open(destination, "wb") as f:
                shutil.copyfileobj(r.raw, f)


def compute_hash(filename):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def compute_hash_benchmark(tgt_folder):
    hashes = []
    for fname in os.listdir(tgt_folder):
        h = compute_hash(Path(tgt_folder) / fname)
        hashes.append(h)
    aggregated_hash = hashlib.sha256()
    [aggregated_hash.update(h.encode('utf-8')) for h in hashes]
    return aggregated_hash.hexdigest()


def compare_hash(tgt_folder, original_hash):
    current_hash = compute_hash_benchmark(tgt_folder)
    return original_hash == current_hash
