import time

import pandas as pd
import numpy as np
from benchmarking.blackbox_repository.conversion_scripts.scripts.lcbench.api import Benchmark
from benchmarking.blackbox_repository.conversion_scripts.utils import repository_path

n_load = 3
n_read = 1000
task = "Fashion-MNIST"

def benchmark_bb_repo(blackbox, task):
    from benchmarking.blackbox_repository import load

    tic_load = time.perf_counter()
    for k in range(n_load):
        bb = load(blackbox)[task]
    toc_load = time.perf_counter()
    bb = load(blackbox)[task]

    tic_read = time.perf_counter()
    hps = np.random.randint(0, 2000, size=n_read)
    for hp in hps:
        # get all fidelities by not passing the fidelity argument
        res = bb(hp)
    toc_read = time.perf_counter()
    return (toc_load - tic_load) / n_load, (toc_read - tic_read) / n_read

def benchmark_lcbench(task):
    tic_load = time.perf_counter()
    for k in range(n_load):
        bench = Benchmark(str(repository_path / "data_2k_lw.json"), cache=True, cache_dir="/tmp/")
    toc_load = time.perf_counter()

    tic_read = time.perf_counter()
    hps = np.random.randint(0, 2000, size=n_read)
    for hp in hps:
        for tag in ["Train/val_accuracy", "time"]:
            res = bench.query(dataset_name=task, tag=tag, config_id=hp)

    toc_read = time.perf_counter()
    return (toc_load - tic_load) / n_load, (toc_read - tic_read) / n_read

rows = []
for method in [
    "HPOBench",
    "Syne Tune",
]:
    if method == 'HPOBench':
        time_load, time_read = benchmark_lcbench(task)
    else:
        time_load, time_read = benchmark_bb_repo("lcbench", task)
    rows.append({
        'blackbox': "lcbench",
        'method': method,
        'load': time_load,
        'read': time_read,
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("results.csv", index=False)
print(df.pivot(index=['method'], columns=['blackbox'], values=['load', 'read']).to_latex())

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