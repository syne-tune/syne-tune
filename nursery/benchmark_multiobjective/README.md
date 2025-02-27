# Benchmark Multiobjective Methods in Syne Tune

This folder shows one way to run quick experiments running different scheduler on different benchmarks and plot
 results once they are done.

To run all experiments, you can run the following:


```bash
pip install -r requirements.txt
pip install -r requirements-moo.txt
python benchmarking/nursery/benchmark_multiobjective/hpo_main.py --experiment_tag "my-new-experiment" --num_seeds 2
```

Which will run all combinations of methods/benchmark/seeds on your local computer for 2 seeds.

You can also run only one scheduler by doing `python benchmarking/nursery/benchmark_multiobjective/hpo_main.py --method RS`, see
`benchmark_main.py` to see all options supported.
