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
from pathlib import Path
from typing import List, Union, Dict, Optional

from syne_tune.try_import import try_import_aws_message, try_import_yahpo_message

try:
    import s3fs as s3fs
    from botocore.exceptions import NoCredentialsError
except ImportError:
    print(try_import_aws_message())

from syne_tune.blackbox_repository.blackbox import Blackbox
from syne_tune.blackbox_repository.blackbox_offline import (
    deserialize as deserialize_offline,
)
from syne_tune.blackbox_repository.blackbox_tabular import (
    deserialize as deserialize_tabular,
)
from syne_tune.blackbox_repository.conversion_scripts.scripts.pd1_import import (
    deserialize as deserialize_pd1,
)

try:
    from syne_tune.blackbox_repository.conversion_scripts.scripts.yahpo_import import (
        instantiate_yahpo,
    )
except ImportError:
    print(try_import_yahpo_message())

# where the blackbox repository is stored on s3
from syne_tune.blackbox_repository.conversion_scripts.recipes import (
    generate_blackbox_recipes,
)
from syne_tune.blackbox_repository.conversion_scripts.utils import (
    repository_path,
    s3_blackbox_folder,
)


def blackbox_list() -> List[str]:
    """
    :return: list of blackboxes available
    """
    return list(generate_blackbox_recipes.keys())


def load_blackbox(
    name: str,
    skip_if_present: bool = True,
    s3_root: Optional[str] = None,
    generate_if_not_found: bool = True,
    yahpo_kwargs: Optional[dict] = None,
) -> Union[Dict[str, Blackbox], Blackbox]:
    """
    :param name: name of a blackbox present in the repository, see blackbox_list() to get list of available blackboxes.
    Syne Tune currently provides the following blackboxes evaluations:
    * "nasbench201": 15625 multi-fidelity configurations of computer vision architectures evaluated on 3 datasets.
    NAS-Bench-201: Extending the scope of reproducible neural architecture search.
    Dong, X. and Yang, Y. 2020.
    * "fcnet": 62208 multi-fidelity configurations of MLP evaluated on 4 datasets.
    Tabular benchmarks for joint architecture and hyperparameter optimization.
    Klein, A. and Hutter, F. 2019.
    * "lcbench": 2000 multi-fidelity Pytorch model configurations evaluated on many datasets.
    Reference: Auto-PyTorch: Multi-Fidelity MetaLearning for Efficient and Robust AutoDL.
    Lucas Zimmer, Marius Lindauer, Frank Hutter. 2020.
    * "icml-deepar": 2420 single-fidelity configurations of DeepAR forecasting algorithm evaluated on 10 datasets.
    A quantile-based approach for hyperparameter transfer learning.
    Salinas, D., Shen, H., and Perrone, V. 2021.
    * "icml-xgboost": 5O00 single-fidelity configurations of XGBoost evaluated on 9 datasets.
    A quantile-based approach for hyperparameter transfer learning.
    Salinas, D., Shen, H., and Perrone, V. 2021.
    * "yahpo-*": Number of different benchmarks from YAHPO Gym. Note that these
        blackboxes come with surrogates already, so no need to wrap them into
        :class:`SurrogateBlackbox`
    :param skip_if_present: skip the download if the file locally exists
    :param s3_root: S3 root directory for blackbox repository. Defaults to
        S3 bucket name of SageMaker session
    :param generate_if_not_found: If the blackbox file is not present locally
        or on S3, should it be generated using its conversion script?
    :param yahpo_kwargs: For a YAHPO blackbox (`name == "yahpo-*"`), these are
        additional arguments to `instantiate_yahpo`
    :return: blackbox with the given name, download it if not present.
    """
    if name.startswith("yahpo-"):
        if yahpo_kwargs is None:
            yahpo_kwargs = dict()
        return instantiate_yahpo(name, **yahpo_kwargs)

    tgt_folder = Path(repository_path) / name
    if (
        tgt_folder.exists()
        and (tgt_folder / "metadata.json").exists()
        and skip_if_present
    ):
        logging.info(
            f"Skipping download of {name} as {tgt_folder} already exists, change skip_if_present to redownload"
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
                "Did not find blackbox files locally nor on S3, regenerating it locally and persisting it on S3."
            )
            generate_blackbox_recipes[name].generate(s3_root=s3_root)

    if name.startswith("pd1"):
        return deserialize_pd1(tgt_folder)
    elif (tgt_folder / "hyperparameters.parquet").exists():
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
        blackbox = load_blackbox(bb)
        print(blackbox)
