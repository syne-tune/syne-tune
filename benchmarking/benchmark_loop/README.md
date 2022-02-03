# Benchmark loop

This folder shows one way to run quick experiments running different scheduler on different benchmarks and plot
 results once they are done.

To run all experiments, you can run the following:


```bash
pip install -r benchmarking/benchmark_loop/requirements.txt
python benchmarking/benchmark_loop/benchmark_main.py --experiment_tag "my-new-experiment" --num_seeds 2
```

Which will run all combinations of methods/benchmark/seeds on your local computer (may take a few hours).

Once all evaluations are done, you can pull results by running:

```python benchmarking/benchmark_loop/plot_results.py --experiment_tag "my-new-experiment"``` 

you will obtain a plot like this one showing the confidence intervals of performance over time:

![alt text](nas201-cifar100.png "Results")

You can also only run only one scheduler by doing `python benchmarking/benchmark_loop/benchmark_main.py --method RS`, see
`benchmark_main.py` to see all options supported.

To launch the evaluation remotely, you can also run 
```python benchmarking/benchmark_loop/launch_remote.py --experiment_tag "my-new-experiment"``` which 
evaluate everything in a remote machine. 

To evaluate other methods/benchmarks, you can edit the following files:
* `baselines.py`: dictionary of HPO methods to be evaluated 
* `benchmark_definitions.py`: dictionary of simulated benchmark to evaluate
* `benchmark_main.py`: script to launch evaluations, run all combinations by default
* `launch_remote.py`: script to launch evaluations on a remote instance
* `plot_results.py`: script to plot results obtained 
* `requirements.txt`: dependencies to be installed when running on a remote machine.