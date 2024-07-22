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
import logging

from syne_tune.blackbox_repository.conversion_scripts.utils import (
    upload_blackbox,
)
from syne_tune.util import catchtime


class BlackboxRecipe:
    def __init__(self, name: str, cite_reference: str):
        """
        Parent class for a blackbox recipe that allows to generate the blackbox data on disk, see ``FCNETRecipe`` or
        ``LCBenchRecipe`` classes for example on how to add a new benchmark.
        :param name: name of the blackbox
        :param cite_reference: name of the paper to be referenced. A message is prompted when generating the blackbox
        to ask the user to cite the relevant paper.
        """
        self.name = name
        self.cite_reference = cite_reference

    def generate(self, upload_on_hub: bool = True):
        """
        Generates the blackbox on disk then upload it on HuggingFace hub
        :return:
        """
        message = (
            f"Generating {self.name} blackbox locally, if you use this dataset in a publication, please cite "
            f'the following paper: "{self.cite_reference}"'
        )
        logging.info(message)
        self._generate_on_disk()

        if upload_on_hub:
            with catchtime(f"Uploading blackbox {self.name} to HuggingFace hub"):
                upload_blackbox(name=self.name)

    def _generate_on_disk(self):
        """
        Method to be overloaded by the child class that should generate the blackbox on disk (handling the donwloading
        and reformatting of external files).
        :return:
        """
        raise NotImplementedError()
