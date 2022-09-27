# Contributing a New Scheduler: Random Search

Random search is arguably the simplest HPO baseline. In a nutshell, `_suggest`
samples a new configuration at random from the configuration space, much
like our `SimpleScheduler` above, and `on_trial_result` does nothing except
returning `SchedulerDecision.CONTINUE`. A slightly more advanced version would
make sure that the same configuration is not suggested twice.

In this section, we walk through the Syne Tune implementation of random search,
thereby discussing some additional concepts. This will also be a first example
of the modular concept just described: random search is implemented as generic
[FIFOScheduler](../../../syne_tune/optimizer/schedulers/fifo.py) configured by a
[RandomSearcher](../../../syne_tune/optimizer/schedulers/searchers/searcher.py).
A self-contained implementation of random search would be shorter. On the other
hand, as seen in [baselines](../../../syne_tune/optimizer/baselines.py),
`FIFOScheduler` also powers GP-based Bayesian optimization, grid search, BORE,
regularized evoluation and constrained BO simply by specifying different
searchers. A number of concepts, to be discussed here, have to be implemented
once only and can be maintained much more easily.


## FIFOScheduler and RandomSearcher

We will have a close look at
[FIFOScheduler](../../../syne_tune/optimizer/schedulers/fifo.py) and
[RandomSearcher](../../../syne_tune/optimizer/schedulers/searchers/searcher.py).
Let us first consider the arguments of `FIFOScheduler`:
* `searcher`, `search_options`: These are used to configure the scheduler with
  a searcher. For ease of use, `searcher` can be a name, and additional
  arguments can be passed via `search_options`. In this case, the searcher is
  created by a factory, as detailed [below](new_searcher.md). Alternatively,
  `searcher` can also be a
  [BaseSearcher](../../../syne_tune/optimizer/schedulers/searchers/searcher.py)
  object.
* `metric`, `mode`: As discussed [above](first_example.md#first-example) in
  `SimpleScheduler`.
* `random_seed`: Several pseudo-random number generators may be used in
  scheduler and searcher. Seeds for these are drawn from a random seed
  generator maintained in `FIFOScheduler`, whose seed can be passed here.
  As a general rule, all schedulers and searchers implemented in Syne Tune
  carefully manage such generators (and contributed schedulers are strongly
  encourage to adopt this pattern).
* `points_to_evaluate`: A list of configurations (possibly partially specified)
  to be suggested first. This allows the user to initialize the search by
  default configurations, thereby injecting knowledge about the task. We
  strongly recommend every scheduler to support this mechanism. More details
  are given below.
* `max_resource_attr`, `max_t`: These arguments are relevant for multi-fidelity
  schedulers. Only one of them needs to be given. We recommend to use
  `max_resource_attr`. More details are given
  [below](extend_async_hb.md#hyperbandscheduler).

The most important use case is to configure `FIFOScheduler` with a new searcher,
and we will concentrate on this one. First, the base class of all searchers is
[BaseSearcher](../../../syne_tune/optimizer/schedulers/searchers/searcher.py):
* `points_to_evaluate`: A list of configurations to be suggested first. This
  is initialized and (possibly) imputed in the base class, but needs to be used
  in child classes. Configurations in `points_to_evaluate` can be partially
  specified. Any hyperparameter missing in a configuration is imputed using a
  "midpoint" rule. For a numerical parameter, this is the middle of the
  range (in linear or log scale). For a categorical parameter, the first value
  is chosen.
  If `points_evaluate` is not given, the default is `[dict()]`: a single
  initial configuration is determined fully by the midpoint rule. In order not
  to use initial configurations at all, the user has to pass
  `points_to_evaluate=[]`. The imputation of configurations is done in the
  base class.
* `configure_scheduler`: Callback function, allows the searcher to configure
  itself depending on the scheduler. It also allows the searcher to reject
  schedulers it is not compatible with. This method is called automatically
  at the beginning of an experiment.
* `get_config`: This method is called by the scheduler in `_suggest`, it
  delegates the suggestion of a configuration for a new trial to the searcher.
* `on_trial_result`: This is called by the scheduler in its own
  `on_trial_result`, also passing the configuration of the current trial.
  If the searcher maintains a surrogate model (for example, based on a
  Gaussian process), it should update its model with `result` data iff
  `update=True`. This is discussed in more detail
  [below](extend_async_hb.md). Note that `on_trial_result` does not return
  anything: decisions on how to proceed with the trial are not done in the
  searcher.
* `register_pending`: Registers one (or more) pending evaluations, which
  are signals to the searcher that a trial has been started and will return
  an observation in the future. This is important in order to avoid redundant
  suggestions in model-based HPO.
* `evaluation_failed`: Called by the scheduler if a trial failed. Default
  searcher reactions are to remove pending evaluations and not to suggest
  the corresponding configuration again. More advanced constrained searchers
  may also try to avoid nearby configurations in the future.
* `cleanup_pending`: Removes all pending evaluations for a trial. This is
  called by the scheduler when a trial terminates.
* `get_state`, `clone_from_state`: Used in order to serialize and
  de-serialize the searcher
* `debug_log`: There is some built-in support for a detailed log, embedded in
  `FIFOScheduler` and the Syne Tune searchers.

Below `BaseSearcher`, there is
[SearcherWithRandomSeed](../../../syne_tune/optimizer/schedulers/searchers/searcher.py),
which should be used by all searchers which make random decisions. It maintains
a PRN generator and provides methods to serialize and de-serialize its state.

Finally, let us walk through
[RandomSearcher](../../../syne_tune/optimizer/schedulers/searchers/searcher.py):
* There are a few features beyond `SimpleScheduler` above. The searcher does
  not suggest the same configuration twice, and also warns if a finite
  configuration space has been exhausted. It also uses
  [HyperparameterRanges](../../../syne_tune/optimizer/schedulers/searchers/utils/hp_ranges.py)
  for random sampling and comparing configurations (to spot duplicates). This
  is a useful helper class, also for encoding configurations as vectors.
  Detecting duplicates is done by a
  [ExclusionList](../../../syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.common.py).
  Finally, `debug_log` is used for diagnostic logs.
* `get_config` first asks for another entry from `points_to_evaluate` by way
  of `_next_initial_config`. It then samples a new configuration at random,
  checking whether the configuration space is not yet exhausted. If successful,
  it also feeds `debug_log`.
* `_update`: This is not needed for random search, but is used here in order
  to feed `debug_log`.


In the [next section](trial_scheduler_api.md), we look at the `TrialScheduler`
API in more detail.
