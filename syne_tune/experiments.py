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
import json
import logging
from datetime import datetime
from json.decoder import JSONDecodeError
from typing import List, Dict, Callable, Optional
import pandas as pd
from dataclasses import dataclass

from syne_tune.constants import ST_TUNER_TIME, ST_TUNER_CREATION_TIMESTAMP
from syne_tune import Tuner
from syne_tune.util import experiment_path, s3_experiment_path
from syne_tune.try_import import try_import_aws_message

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print(try_import_aws_message())


@dataclass
class ExperimentResult:
    name: str
    results: pd.DataFrame
    metadata: Dict
    tuner: Tuner

    def __str__(self):
        res = f"Experiment {self.name}"
        if self.results is not None:
            res += f" contains {len(self.results)} evaluations from {len(self.results.trial_id.unique())} trials"
        return res

    def creation_date(self):
        return datetime.fromtimestamp(self.metadata[ST_TUNER_CREATION_TIMESTAMP])

    def plot(self, **plt_kwargs):
        import matplotlib.pyplot as plt

        metric = self.metric_names()[0]
        df = self.results
        if df is not None and len(df) > 0:
            df = df.sort_values(ST_TUNER_TIME)
            x = df.loc[:, ST_TUNER_TIME]
            y = (
                df.loc[:, metric].cummax()
                if self.metric_mode() == "max"
                else df.loc[:, metric].cummin()
            )
            plt.plot(x, y, **plt_kwargs)
            plt.xlabel("wallclock time")
            plt.ylabel(metric)
            plt.title(f"Best result over time {self.name}")
            plt.legend()
            plt.show()

    def metric_mode(self) -> str:
        return self.metadata["metric_mode"]

    def metric_names(self) -> List[str]:
        return self.metadata["metric_names"]

    def entrypoint_name(self) -> str:
        return self.metadata["entrypoint"]

    def best_config(self) -> Dict:
        """
        Return the best config found for the first metric defined in the scheduler.
        :param self:
        :return:
        """
        metric_names = self.metric_names()
        metric_mode = self.metric_mode()

        if len(metric_names) > 1:
            logging.warning(
                "Several metrics exists so the best is not defined, this will return the best other the"
                f"first metric {metric_names}."
            )
        metric_name = metric_names[0]

        # locate best result
        if metric_mode == "min":
            best_index = self.results.loc[:, metric_name].argmin()
        else:
            best_index = self.results.loc[:, metric_name].argmax()
        res = dict(self.results.loc[best_index])

        # dont include internal fields
        return {k: v for k, v in res.items() if not k.startswith("st_")}


def download_single_experiment(
    tuner_name: str,
    s3_bucket: Optional[str] = None,
    experiment_name: Optional[str] = None,
):
    """
    Downloads results from s3 of a tuning experiment previously run with remote launcher.
    :param tuner_name: named of the tuner to be retrieved.
    :param s3_bucket: If not given, the default bucket for the SageMaker session is used
    :param experiment_name: If given, this is used as first directory.
    :return:
    """
    s3_path = s3_experiment_path(
        s3_bucket=s3_bucket, tuner_name=tuner_name, experiment_name=experiment_name
    ).rstrip("/")
    tgt_dir = experiment_path(tuner_name=tuner_name)
    tgt_dir.mkdir(exist_ok=True, parents=True)
    s3 = boto3.client("s3")
    parts_path = s3_path.replace("s3://", "").split("/")
    s3_bucket = parts_path[0]
    s3_key = "/".join(parts_path[1:])
    for file in ["metadata.json", "results.csv.zip", "tuner.dill"]:
        try:
            logging.info(f"downloading {file} on {s3_path}")
            s3.download_file(s3_bucket, f"{s3_key}/{file}", str(tgt_dir / file))
        except ClientError as e:
            logging.info(f"could not find {file} on {s3_path}")


def load_experiment(
    tuner_name: str,
    download_if_not_found: bool = True,
    load_tuner: bool = False,
    local_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> ExperimentResult:
    """
    :param tuner_name: name of a tuning experiment previously run
    :param download_if_not_found: whether to fetch the experiment from s3 if not found locally
    :param load_tuner: whether to load the tuner in addition to metadata and results
    :param local_path: path containing the experiment to load if not specified, then `~/{SYNE_TUNE_FOLDER}/` is used.
    :return:
    """
    path = experiment_path(tuner_name, local_path)

    metadata_path = path / "metadata.json"
    if not (metadata_path.exists()) and download_if_not_found:
        logging.info(
            f"experiment {tuner_name} not found locally, trying to get it from s3."
        )
        download_single_experiment(
            tuner_name=tuner_name, experiment_name=experiment_name
        )
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = None
    try:
        if (path / "results.csv.zip").exists():
            results = pd.read_csv(path / "results.csv.zip")
        else:
            results = pd.read_csv(path / "results.csv")
    except Exception:
        results = None
    if load_tuner:
        try:
            tuner = Tuner.load(path)
        except FileNotFoundError:
            tuner = None
        except Exception:
            tuner = None
    else:
        tuner = None
    return ExperimentResult(
        name=tuner.name if tuner is not None else path.stem,
        results=results,
        tuner=tuner,
        metadata=metadata,
    )


def get_metadata(
    name_filter: Callable[[str], bool] = None, root=experiment_path()
) -> Dict[str, Dict]:
    """
    :param name_filter: if passed then only experiments whose path matching the filter are kept. This allows
    rapid filtering in the presence of many experiments.
    :return: dictionary from tuner name to metadata dict
    """
    res = {}
    for metadata_path in root.glob("**/metadata.json"):
        path = metadata_path.parent
        if name_filter is None or name_filter(str(path)):
            try:
                tuner_name = path.name
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    # we check that the metadata is valid by verifying that is a dict containing Syne Tune time-stamp
                    if (
                        isinstance(metadata, Dict)
                        and ST_TUNER_CREATION_TIMESTAMP in metadata
                    ):
                        metadata["path"] = str(path.parent)
                        res[tuner_name] = metadata
            except JSONDecodeError as e:
                print(f"could not read {path}")
                pass
    return res


def list_experiments(
    path_filter: Callable[[str], bool] = None,
    experiment_filter: Callable[[ExperimentResult], bool] = None,
    load_tuner: bool = False,
) -> List[ExperimentResult]:
    res = []
    for metadata_path in experiment_path().glob("**/metadata.json"):
        path = metadata_path.parent
        tuner_name = path.name
        if path_filter is None or path_filter(metadata_path):
            exp = load_experiment(tuner_name, load_tuner, local_path=path.parent)
            if experiment_filter is None or experiment_filter(exp):
                if exp.results is not None and exp.metadata is not None:
                    res.append(exp)
    return sorted(
        res,
        key=lambda exp: exp.metadata.get(ST_TUNER_CREATION_TIMESTAMP, 0),
        reverse=True,
    )


def load_experiments_df(
    path_filter: Callable[[str], bool] = None,
    experiment_filter: Callable[[ExperimentResult], bool] = None,
    load_tuner: bool = False,
) -> pd.DataFrame:
    """
    :param: name_filter: if specified, only experiment whose path name matches the filter will be kept.
    :param experiment_filter: only experiment where the filter is True are kept, default to None and returns everything.
    :return: a dataframe that contains all evaluations reported by tuners according to the filter given.
    The columns contains trial-id, hyperparameter evaluated, metrics observed by `report`:
     metrics collected automatically by syne-tune:
     `st_worker_time` (indicating time spent in the worker when report was seen)
     `time` (indicating wallclock time measured by the tuner)
     `decision` decision taken by the scheduler when observing the result
     `status` status of the trial that was shown to the tuner
     `config_{xx}` configuration value for the hyperparameter {xx}
     `tuner_name` named passed when instantiating the Tuner
     `entry_point_name`/`entry_point_path` name and path of the entry point that was tuned
    """
    dfs = []
    for experiment in list_experiments(
        path_filter=path_filter,
        experiment_filter=experiment_filter,
        load_tuner=load_tuner,
    ):
        assert experiment.results is not None
        assert experiment.metadata is not None

        df = experiment.results
        df["tuner_name"] = experiment.name
        for k, v in experiment.metadata.items():
            if isinstance(v, List):
                if len(v) > 1:
                    for i, x in enumerate(v):
                        df[f"{k}-{i}"] = x
                else:
                    df[k] = v[0]
            else:
                df[k] = v
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    for exp in list_experiments():
        if exp.results is not None:
            print(exp)
