# Benchmarking

This repository shows an example on how to run a full evaluation: launching tuning experiments and plotting results.

## How to run it

**Local Machine.**

Setup an environment by doing
```
pip install -r benchmarking/requirements.txt
```

Then run
```
python benchmarking/benchmark_main.py --seed 0
python benchmarking/benchmark_main.py --seed 1 
python benchmarking/benchmark_main.py --seed 2  
```
which will evaluate all methods on all blackboxes for 3 seeds.


**Slurm.** You can also run on Slurm, for this you need to first install [Slurmpilot](https://github.com/geoalgo/slurmpilot/tree/main) and setup your cluster.

Then you can do:

```bash
python benchmarking/launch_slurmpilot.py --cluster YOURCLUSTER --partition YOURPARTITION --num_seeds 3
```

After your results are done, you can download your results with
```bash
sp --download YOURJOBNAME
```

Note the path where the data is downloaded, you can then plot results with: 
```
pip install pyparfor
python benchmarking/results_analysis/show_results.py --path "~/slurmpilot/jobs/synetune/bench-2025-02-20-16-00-38/results" 
```
after adapting `path` to where the slurmpilot data was downloaded.

## Contributing

Right now, the files are meant to be copy-pasted so if want to run an experiments (unless you just
want to do an evaluation with the current method/dataset available).

We are currently looking only for contributions which adds methods or benchmarks.