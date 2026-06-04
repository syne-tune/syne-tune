# Benchmarking

This repository shows an example of how to run a full evaluation: launching tuning experiments and plotting results.

## How to run it

**Local Machine.**

Setup an environment by doing
```bash
pip install -r benchmarking/requirements.txt
```

Then run
```bash
python benchmarking/benchmark_main.py --seed 0
python benchmarking/benchmark_main.py --seed 1 
python benchmarking/benchmark_main.py --seed 2  
```
which will evaluate all methods on all blackboxes for 3 seeds.

You can also run a specific method on a specific benchmark. For example:
```bash
python benchmarking/benchmark_main.py --seed 0 --method RS --benchmark fcnet-protein
```
To see all available methods and benchmarks, check `benchmarking/baselines.py` and `benchmarking/benchmarks.py`.

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
```bash
pip install pyparfor
python benchmarking/results_analysis/show_results.py --path "~/slurmpilot/jobs/synetune/bench-2025-02-20-16-00-38/results" 
```
after adapting `--path` to where the slurmpilot data was downloaded.

## How to run a custom method on existing benchmarks

To evaluate your own custom optimization method against existing baselines, you first need to define your searcher and then register it.

1. **Define a Custom Searcher**: Create a class inheriting from `SingleObjectiveBaseSearcher` and implement the required methods.
```python
from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import SingleObjectiveBaseSearcher

class MyCustomSearcher(SingleObjectiveBaseSearcher):
    def __init__(
        self,
        config_space: dict,
        metric: str,
        points_to_evaluate: list[dict] | None = None,
        random_seed: int | None = None,
    ):
        super().__init__(
            config_space=config_space, 
            points_to_evaluate=points_to_evaluate, 
            random_seed=random_seed
        )
        self.metric = metric
        # Initialize your surrogate model or tracking state here
        self.results = []

    def suggest(self, **kwargs) -> dict | None:
        # First, query points_to_evaluate to evaluate the initial points
        config = self._next_points_to_evaluate()
        if config is not None:
            return config
            
        # Write your custom sampling/optimization logic here.
        # For example, simply sampling at random:
        return {
            k: v.sample() if hasattr(v, "sample") else v
            for k, v in self.config_space.items()
        }
        
    def on_trial_result(self, trial_id: int, config: dict, metric: float):
        # Update your surrogate model or state with intermediate results
        pass

    def on_trial_complete(self, trial_id: int, config: dict, metric: float):
        # Update your surrogate model or state with final results
        self.results.append((config, metric))
```

2. **Register the Method**: Open `benchmarking/baselines.py` and add a new entry to the `methods` dictionary. 
The dictionary key is the name of the method (e.g., `"MyCustomMethod"`) and the value is a callable that takes a `MethodArguments` object and returns a `TrialScheduler`.

```python
from syne_tune.optimizer.schedulers.single_objective_scheduler import SingleObjectiveScheduler
from benchmarking.baselines import methods, MethodArguments
# Import MyCustomSearcher from where you defined it

methods["MyCustomMethod"] = lambda method_arguments: SingleObjectiveScheduler(
    config_space=method_arguments.config_space,
    searcher=MyCustomSearcher(
        config_space=method_arguments.config_space, 
        metric=method_arguments.metric,
        points_to_evaluate=method_arguments.points_to_evaluate,
        random_seed=method_arguments.random_seed
    ),
    metric=method_arguments.metric,
    do_minimize=method_arguments.mode == "min",
    random_seed=method_arguments.random_seed,
)
```

After adding your method, you can run an evaluation with it from the CLI:
```bash
python benchmarking/benchmark_main.py --seed 0 --method MyCustomMethod --benchmark fcnet-protein
```

## Contributing

We are currently looking primarily for contributions which add methods or benchmarks. 
