import logging
from typing import List, Union, Dict, Optional

from huggingface_hub import snapshot_download

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

from syne_tune.blackbox_repository.conversion_scripts.scripts.hpob_import import (
    deserialize as deserialize_hpob,
)

from syne_tune.blackbox_repository.conversion_scripts.recipes import (
    generate_blackbox_recipes,
)
from syne_tune.blackbox_repository.conversion_scripts.utils import (
    repository_path,
    repo_id,
)

logger = logging.getLogger(__name__)


def blackbox_list() -> List[str]:
    """
    :return: list of blackboxes available
    """
    return list(generate_blackbox_recipes.keys())


def load_blackbox(
    name: str,
    custom_repo_id: Optional[str] = None,
    yahpo_kwargs: Optional[dict] = None,
    local_files_only: bool = False,
    force_download: bool = False,
    **snapshot_download_kwargs,
) -> Union[Dict[str, Blackbox], Blackbox]:
    """
    :param name: name of a blackbox present in the repository, see
        :func:`blackbox_list` to get list of available blackboxes. Syne Tune
        currently provides the following blackboxes evaluations:

        * "nasbench201": 15625 multi-fidelity configurations of computer vision
          architectures evaluated on 3 datasets.
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
        * "pd1": 23 multi-fidelity benchmarks for hyperparameter optimization of neural networks for image classification
          Pre-trained Gaussian processes for Bayesian optimization.
          Wang, Z. and Dahl G. and Swersky K. and Lee C. and Nado Z. and Gilmer J. and Snoek J. and Ghahramani Z. 2021.
        * "icml-xgboost": 5O00 single-fidelity configurations of XGBoost evaluated on 9 datasets.
          A quantile-based approach for hyperparameter transfer learning.
          Salinas, D., Shen, H., and Perrone, V. 2021.
        * "yahpo-*": Number of different benchmarks from YAHPO Gym. Note that these
          blackboxes come with surrogates already, so no need to wrap them into
          :class:`SurrogateBlackbox`
        * "hpob_*": ca. 6.34 million evaluations distributed on 16 search spaces and 101 datasets.
          HPO-B: A Large-Scale Reproducible Benchmark for Black-Box HPO based on OpenML.
          S. Arango, H. Jomaa, M. Wistuba, J. Grabocka, 2021.
        * "tabrepo-*": TabRepo contains the predictions and metrics of 1530 models evaluated on 211 classification and regression datasets.
          TabRepo: A Large Scale Repository of Tabular Model Evaluations and its AutoML Applications.
          D. Salinas, N. Erickson, 2024.
    :param custom_repo_id: custom hugging face repoid to use, default to Syne Tune hub
    :param yahpo_kwargs: For a YAHPO blackbox (``name == "yahpo-*"``), these are
        additional arguments to ``instantiate_yahpo``
    :param local_files_only: whether to use local files with no internet check on the Hub
    :param force_download: forces files to be downloaded
    :param snapshot_download_kwargs: keyword arguments for `snapshot_download` (other than local_files_only and force_download)
    :return: blackbox with the given name, download it if not present.
    """
    assert (
        name in blackbox_list()
    ), f"Got {name} but only the following blackboxes are supported {blackbox_list()}."

    # download blackbox if not present, we use allow_patterns to download only the files wanted
    if not name.startswith("yahpo"):
        allow_patterns = f"{name}/*"
    else:
        allow_patterns = f"yahpo/*"

    snapshot_download(
        repo_id=custom_repo_id if custom_repo_id else repo_id,
        repo_type="dataset",
        # for now we use allow_pattern for lack of a better option to specify explicitly the desired blackbox directory
        allow_patterns=allow_patterns,
        local_dir=repository_path,
        force_download=force_download,
        local_files_only=local_files_only,
        **snapshot_download_kwargs,
    )

    # TODO avoid switch case of PD1
    blackbox_path = repository_path / name
    if name.startswith("yahpo"):
        from syne_tune.blackbox_repository.conversion_scripts.scripts.yahpo_import import (
            instantiate_yahpo,
        )

        if yahpo_kwargs is None:
            yahpo_kwargs = dict()
        return instantiate_yahpo(name, **yahpo_kwargs)
    elif name.startswith("pd1"):
        return deserialize_pd1(blackbox_path)
    elif name.startswith("hpob"):
        return deserialize_hpob(blackbox_path)
    elif (blackbox_path / "hyperparameters.parquet").exists():
        return deserialize_tabular(blackbox_path)
    else:
        return deserialize_offline(blackbox_path)


def check_blackbox_local_files(tgt_folder) -> bool:
    """checks whether the file of the blackbox ``name`` are present in ``repository_path``"""
    return tgt_folder.exists() and (tgt_folder / "metadata.json").exists()


if __name__ == "__main__":
    # list all blackboxes available
    blackboxes = blackbox_list()
    print(blackboxes)

    for bb in blackboxes:
        print(bb)
        # download an existing blackbox
        blackbox = load_blackbox(bb)
        print(blackbox)
