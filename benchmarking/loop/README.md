# Benchmarking

## How to run it

**Local Machine.**
Setup an environment by doing
```
pip install -r benchmarking/nursery/benchmark_conformal/requirements.txt
```

Then run
```
python benchmarking/nursery/benchmark_main.py --num_seeds 3 
```
which will evaluate all methods on all blackboxes for 3 seeds.


**Slurm.** You can also run on Slurm, for this you need to first install Slurmpilot and setup your cluster.

Then you can do:

```bash
python benchmarking/nursery/loop/launch_slurmpilot.py --cluster YOURCLUSTER --partition YOURPARTITION --num_seeds 3
```

After your results are done, you can download your results with
```bash
sp --download YOURJOBNAME
```

Note the path where the data is downloaded, you can then plot results with: 
```
python benchmarking/loop/results_analysis/show_results.py --path "~/slurmpilot/jobs/synetune/bench-2025-02-20-16-00-38/results" 
```
after adapting `path` to where the slurmpilot data was downloaded.