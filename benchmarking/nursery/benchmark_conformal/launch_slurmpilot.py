import os

from argparse import ArgumentParser

from pathlib import Path

from coolname import generate_slug
from tqdm import tqdm

from syne_tune.util import random_string

from slurmpilot.config import load_config
from slurmpilot.slurm_wrapper import SlurmWrapper, JobCreationInfo
from slurmpilot.util import unify

from baselines import (
    methods,
    Methods,
)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag", type=str, required=False, default=generate_slug(2)
    )
    parser.add_argument(
        "--experiment_subtag", type=str, required=False, default=random_string(4)
    )
    parser.add_argument("--n_workers", type=int, required=False, default=1)
    parser.add_argument("--num_seeds", type=int, required=False, default=1)
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag
    experiment_subtag = args.experiment_subtag
    num_seeds = args.num_seeds

    print(args)
    print(f"Methods defined: {list(methods.keys())}")
    methods_selected = [
        Methods.RS,
        Methods.LegacyRS,
        Methods.REA,
        Methods.LegacyREA,
        Methods.TPE,
        Methods.LegacyTPE,
        Methods.GP,
        Methods.BORE,
        Methods.LegacyBORE,
        Methods.BOTorch,
        Methods.LegacyBOTorch,
        Methods.CQR,
        Methods.LegacyCQR,
        # MF
        Methods.ASHA,
        Methods.LegacyASHA,
        Methods.ASHABORE,
        Methods.LegacyASHABORE,
        Methods.ASHACQR,
        Methods.LegacyASHACQR,
        Methods.BOHB,
        Methods.LegacyBOHB,
        Methods.MOBSTER,
        Methods.HYPERTUNE,
    ]
    print(f"{len(methods_selected)} methods selected: {methods_selected}")

    config = load_config()
    cluster, partition = "kislurm", ""
    #    cluster, partition = 'scule', 'paul'

    slurm = SlurmWrapper(config=config, clusters=[cluster], ssh_engine="ssh")
    max_runtime_minutes = 60 * 24
    python_args = []
    for method in tqdm(methods_selected):
        assert method in methods
        python_args.append(
            {
                "experiment_tag": experiment_tag,
                "subtag": experiment_subtag,
                "method": method,
                "n_workers": args.n_workers,
                "num_seeds": num_seeds,
            }
        )

    jobinfo = JobCreationInfo(
        cluster=cluster,
        partition=partition,
        jobname=unify("syne-tune/benchmarking", method="coolname"),
        entrypoint="benchmark_main.py",
        python_args=python_args,
        src_dir=str(Path(__file__).parent),
        # python_libraries=["/home/aaron/git/syne-tune/syne_tune"],
        python_binary="python",
        n_cpus=1,
        mem=1024 * 8,
        max_runtime_minutes=60 * 24,
        bash_setup_command='eval "$(conda shell.bash hook)" && conda activate syne_tune',
        env={
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        },
    )
    jobid = slurm.schedule_job(jobinfo)
