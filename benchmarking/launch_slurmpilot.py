import os
from argparse import ArgumentParser
from pathlib import Path

from coolname import generate_slug
from slurmpilot.config import load_config
from slurmpilot.slurm_wrapper import SlurmWrapper, JobCreationInfo
from slurmpilot.util import unify
from tqdm import tqdm

from baselines import (
    methods,
    Methods,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_tag", type=str, required=False, default="bench")
    parser.add_argument("--n_workers", type=int, required=False, default=1)
    parser.add_argument("--num_seeds", type=int, required=False, default=1)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--cluster", type=str, required=True)
    parser.add_argument("--partition", type=str, required=True)
    parser.add_argument(
        "--slurmpilot_folder", type=str, required=False, default="~/slurmpilot/jobs"
    )

    parser.add_argument("--sbatch_arguments", type=str, required=False)

    args, _ = parser.parse_known_args()

    experiment_tag = args.experiment_tag

    num_seeds = args.num_seeds
    cluster = args.cluster
    partition = args.partition
    sbatch_arguments = args.sbatch_arguments
    slurmpilot_folder = args.slurmpilot_folder

    print(args)
    print(f"Methods defined: {list(methods.keys())}")
    methods_selected = [
        Methods.LegacyRS,
        # Methods.LegacyREA,
        Methods.LegacyTPE,
        Methods.LegacyBORE,
        Methods.LegacyCQR,
        # Methods.LegacyBOTorch,
        Methods.LegacyASHA,
        Methods.LegacyASHABORE,
        Methods.LegacyASHACQR,
        # Methods.LegacyBOHB,
        Methods.BORE,
        Methods.TPE,
        Methods.CQR,
        Methods.ASHACQR,
        Methods.ASHABORE,
        Methods.BOHB,
    ]
    print(f"{len(methods_selected)} methods selected: {methods_selected}")

    config = load_config()
    #    cluster, partition = 'scule', 'paul'

    slurm = SlurmWrapper(config=config, clusters=[cluster], ssh_engine="ssh")
    max_runtime_minutes = 60 * 4
    python_args = []
    for method in tqdm(methods_selected):
        assert method in methods
        for seed in range(num_seeds):
            python_args.append(
                {
                    "method": method,
                    "n_workers": args.n_workers,
                    # TODO this runs only a given seed, the API of benchmark_main is quite ugly
                    #  we could simplify this by only supporting `--seed N` as argument to benchmark_main.py
                    "seed": seed,
                }
            )

    jobname = unify(f"synetune/{experiment_tag}", method="date")
    print(f"Going to launch {len(python_args)} jobs, jobname: {jobname}")
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
    jobinfo = JobCreationInfo(
        cluster=cluster,
        partition=partition,
        sbatch_arguments=sbatch_arguments,
        jobname=jobname,
        entrypoint="benchmark_main.py",
        python_args=python_args,
        src_dir=str(Path(__file__).parent),
        python_binary="python",
        python_libraries=[
            str(Path(__file__).parent.parent / "syne_tune"),
        ],
        n_cpus=8,
        mem=1024 * 8,
        max_runtime_minutes=max_runtime_minutes,
        bash_setup_command="source ~/.bashrc; conda activate syne_tune",
        env={
            # write tuner files in Slurmpilot folder corresponding to `jobname`
            "SYNETUNE_FOLDER": f"{slurmpilot_folder}/{jobname}",
        },
        n_concurrent_jobs=128,  # max number of jobs to run at the same time
    )
    if not args.dry_run:
        jobid = slurm.schedule_job(jobinfo)
