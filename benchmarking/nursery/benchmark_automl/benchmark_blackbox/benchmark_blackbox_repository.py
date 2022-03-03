import time

import pandas as pd


n_load = 3
n_read = 1000

def benchmark_bb_repo(blackbox, task):
    from benchmarking.blackbox_repository import load

    tic_load = time.perf_counter()
    for k in range(n_load):
        bb = load(blackbox)[task]
    toc_load = time.perf_counter()

    tic_read = time.perf_counter()
    config_space = bb.configuration_space
    for k in range(n_read):
        hp = {key: v.sample() for key, v in config_space.items()}
        # get all fidelities by not passing the fidelity argument
        bb(hp)
    toc_read = time.perf_counter()
    return (toc_load - tic_load) / n_load, (toc_read - tic_read) / n_read

def benchmark_hpobench(blackbox="fcnet", task=None):
    from hpobench.benchmarks.nas.nasbench_201 import Cifar100NasBench201Benchmark
    from hpobench.container.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark
    tic_load = time.perf_counter()
    for k in range(n_load):
        if blackbox == "fcnet":
            bb = SliceLocalizationBenchmark(task_id=167149, rng=1)
        else:
            bb = Cifar100NasBench201Benchmark(task_id=167149, rng=1)
    toc_load = time.perf_counter()

    tic_read = time.perf_counter()
    fidelity_size = int(bb.get_fidelity_space().get_hyperparameters()[0].get_size())
    fidelity_name = bb.get_fidelity_space().get_hyperparameters()[0].name
    for k in range(n_read):
        hp = bb.get_configuration_space().sample_configuration()
        for fidelity in range(fidelity_size):
            bb.objective_function(hp, {fidelity_name: fidelity + 1})
    toc_read = time.perf_counter()
    return (toc_load - tic_load) / n_load, (toc_read - tic_read) / n_read

rows = []
for blackbox, task in [("fcnet", "slice_localization"), ("nasbench201", "cifar100")]:
    for method in ["HPOBench", "Syne Tune"]:
        if method == 'HPOBench':
            time_load, time_read = benchmark_hpobench(blackbox, task)
        else:
            time_load, time_read = benchmark_bb_repo(blackbox, task)
        rows.append({
            'blackbox': blackbox,
            'method': method,
            'load': time_load,
            'read': time_read,
        })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("results.csv", index=False)
print(df.pivot(index=['method'], columns=['blackbox'], values=['load', 'read']).to_latex(float_format="%.4f"))

"""
\begin{tabular}{lrrrr}
\toprule
{} & \multicolumn{2}{l}{load} & \multicolumn{2}{l}{read} \\
blackbox & fcnet & nasbench201 & fcnet & nasbench201 \\
method    &       &             &       &             \\
\midrule
HPOBench  &  7.90 &       31.55 &  0.00 &        0.00 \\
Syne Tune &  0.94 &       10.95 &  0.00 &        0.00 \\
\bottomrule
\end{tabular}
"""