import logging
from typing import Optional

from syne_tune.util import catchtime


class BlackboxRecipe:
    def __init__(self, name: str, cite_reference: str):
        """
        Parent class for a blackbox recipe that allows to generate the blackbox data on disk, see `FCNETRecipe` or
        `LCBenchRecipe` classes for example on how to add a new benchmark.
        :param name: name of the blackbox
        :param cite_reference: name of the paper to be referenced. A message is prompted when generating the blackbox
        to ask the user to cite the relevant paper.
        """
        self.name = name
        self.cite_reference = cite_reference

    def generate(self, s3_root: Optional[str] = None):
        """
        Generates the blackbox on disk then upload it on s3 if AWS is available.
        :param s3_root: s3 root where to upload to s3, default to s3://{sagemaker-bucket}/blackbox-repository.
        If AWS is not available, this step is skipped and the dataset is just persisted locally.
        :return:
        """
        message = (
            f"Generating {self.name} blackbox locally, if you use this dataset in a publication, please cite "
            f'the following paper: "{self.cite_reference}"'
        )
        logging.info(message)
        self._generate_on_disk()

        with catchtime("uploading to s3"):
            from syne_tune.blackbox_repository.conversion_scripts.utils import (
                upload_blackbox,
            )

            upload_blackbox(name=self.name, s3_root=s3_root)

    def _generate_on_disk(self):
        """
        Method to be overloaded by the child class that should generate the blackbox on disk (handling the donwloading
        and reformatting of external files).
        :return:
        """
        raise NotImplementedError()
