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
    parser.add_argument(
        "--max_num_trials",
        type=int,
        required=False,
        default=None,
        help="If set, each run stops after this number of trials instead of using "
        "the time and evaluation limits of the benchmark.",
    )

    parser.add_argument(
        "--methods",
        type=str,
        required=False,
        default=None,
        help='Space-separated list of methods to run, for instance "CQR RS BOTorch". '
        "Defaults to the methods listed in this script.",
    )

    parser.add_argument("--cluster", type=str, required=True)
    parser.add_argument("--partition", type=str, required=True)
    parser.add_argument(
        "--slurmpilot_folder", type=str, required=False, default="~/slurmpilot/jobs"
    )

    parser.add_argument("--sbatch_arguments", type=str, required=False)
    parser.add_argument(
        "--python_binary",
        type=str,
        required=False,
        default="python",
        help="Python binary used to run the jobs on the cluster.",
    )
    parser.add_argument(
        "--bash_setup_command",
        type=str,
        required=False,
        default="source ~/.bashrc; conda activate synetune",
        help="Bash command run before each job, for instance to activate an environment.",
    )
    parser.add_argument("--n_cpus", type=int, required=False, default=8)
    parser.add_argument(
        "--mem", type=int, required=False, default=8, help="Memory per job in GB."
    )
    parser.add_argument(
        "--env",
        type=str,
        nargs="*",
        required=False,
        default=[],
        help="Additional environment variables for the jobs given as KEY=VALUE, "
        "for instance HF_HUB_OFFLINE=1.",
    )

    args, _ = parser.parse_known_args()

    experiment_tag = args.experiment_tag

    num_seeds = args.num_seeds
    cluster = args.cluster
    partition = args.partition
    sbatch_arguments = args.sbatch_arguments
    slurmpilot_folder = args.slurmpilot_folder
    env = {}
    for key_value in args.env:
        assert "=" in key_value, f"--env expects KEY=VALUE, got {key_value}"
        key, value = key_value.split("=", 1)
        env[key] = value

    print(f"Methods defined: {list(methods.keys())}")
    methods_selected = [
        Methods.RS,
        Methods.REA,
        Methods.TPE,
        Methods.BORE,
        Methods.CQR,
        Methods.BOTorch,
        Methods.BOHB,
        Methods.ASHA,
        Methods.ASHACQR,
        Methods.ASHABORE,
    ]
    if args.methods is not None:
        methods_selected = args.methods.split()
    print(f"{len(methods_selected)} methods selected: {methods_selected}")

    benchmarks_selected = list(benchmark_definitions.keys())
    config = load_config()

    slurm = SlurmPilot(config=config, clusters=[cluster], ssh_engine="ssh")
    max_runtime_minutes = 60 * 24 - 1
    python_args = []
    for method in methods_selected:
        assert method in methods, f"{method} not in {list(methods.keys())}"
    for method in tqdm(methods_selected):
        for benchmark in benchmarks_selected:
            python_args.append(
                {
                    "method": method,
                    "n_workers": args.n_workers,
                    "benchmark": benchmark,
                    # run all seeds in [0, seed-1]
                    "seed": num_seeds,
                    "run_all_seed": 1,
                    **(
                        {"max_num_trials": args.max_num_trials}
                        if args.max_num_trials is not None
                        else {}
                    ),
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
        python_binary=args.python_binary,
        python_libraries=[
            str(Path(__file__).parent.parent / "syne_tune"),
        ],
        n_cpus=args.n_cpus,
        mem=1024 * args.mem,
        max_runtime_minutes=max_runtime_minutes,
        bash_setup_command=args.bash_setup_command,
        env={
            # write tuner files in Slurmpilot folder corresponding to `jobname`
            "SYNETUNE_FOLDER": f"{slurmpilot_folder}/{jobname}",
            **env,
        },
        n_concurrent_jobs=128,  # max number of jobs to run at the same time
    )
    if not args.dry_run:
        jobid = slurm.schedule_job(jobinfo)
