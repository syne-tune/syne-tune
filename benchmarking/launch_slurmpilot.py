from argparse import ArgumentParser
from pathlib import Path

from slurmpilot import SlurmPilot, JobCreationInfo
from slurmpilot.config import load_config
from slurmpilot.util import unify
from tqdm import tqdm

from baselines import (
    methods,
    Methods,
)
from benchmarking.benchmarks import benchmark_definitions

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

    print(f"Methods defined: {list(methods.keys())}")
    methods_selected = [
        Methods.RS,
        Methods.REA,
        Methods.TPE,
        Methods.BORE,
        Methods.CQR,
        Methods.BOTorch,
        # Methods.BOHB,
        # Methods.ASHA,
        # Methods.ASHACQR,
        # Methods.ASHABORE,
    ]
    print(f"{len(methods_selected)} methods selected: {methods_selected}")

    benchmarks_selected = list(benchmark_definitions.keys())
    config = load_config()

    slurm = SlurmPilot(config=config, clusters=[cluster], ssh_engine="ssh")
    max_runtime_minutes = 60 * 24 - 1
    python_args = []
    for method in tqdm(methods_selected):
        assert method in methods, f"{method} not in {methods}"
        for benchmark in benchmarks_selected:
            python_args.append(
                {
                    "method": method,
                    "n_workers": args.n_workers,
                    "benchmark": benchmark,
                    # run all seeds in [0, seed-1]
                    "seed": num_seeds,
                    "run_all_seed": 1,
                }
            )

    jobname = unify(f"synetune/{experiment_tag}", method="date")

    print(f"Going to launch {len(python_args)} jobs, jobname: {jobname}")
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
        bash_setup_command="source ~/.bashrc; conda activate synetune",
        env={
            # write tuner files in Slurmpilot folder corresponding to `jobname`
            "SYNETUNE_FOLDER": f"{slurmpilot_folder}/{jobname}",
        },
        n_concurrent_jobs=128,  # max number of jobs to run at the same time
    )
    if not args.dry_run:
        jobid = slurm.schedule_job(jobinfo)
