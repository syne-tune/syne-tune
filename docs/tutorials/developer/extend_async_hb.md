# Contributing a New Scheduler: Extending Asynchronous Hyperband

Syne Tune provides powerful generic scheduler templates for popular methods
like successive halving and Hyperband. These can be run with synchronous or
asynchronous decision-making. The most important generic templates at the
moment are:
* [FIFOScheduler](random_search.md#fifoscheduler-and-randomsearcher):
  "Full evaluation" scheduler, baseclass for many others.
* [HyperbandScheduler](extend_async_hb.md#hyperbandscheduler):
  Asynchronous successive halving and Hyperband.
* [SynchronousHyperbandScheduler](extend_sync_hb.md#synchronous-hyperband):
  Synchronous successive halving and Hyperband.

Chances are your idea for a new scheduler maps to one of these templates, in
which case you can save a lot of time and headache by just extending the
template, rather than re-implementing the wheel. Due to Syne Tune's modular
design of schedulers and their components (e.g., searchers, decision rules), you
may even get more than you bargained for.

In this section, we will walk through an example of how to furnish the
asynchronous successive halving scheduler with a specific searcher.


## HyperbandScheduler

Details about asynchronous successive halving and Hyperband are given in
the [Multi-fidelity HPO tutorial](../multifidelity/README.md). This is a
multi-fidelity scheduler, where trials report intermediate results (e.g.,
validation error at the end of each epoch of training). We can formalize this
notion by the concept of *resource* $r = 1, 2, 3, \dots$ (e.g., $r$ is the
number of epochs trained). A generic implementation of this method is provided
in [HyperbandScheduler](../../../syne_tune/optimizer/schedulers/hyperband.py#L147).
Let us have a look at its arguments not shared with the base class
`FIFOScheduler`:
* A mandatory argument is `resource_attr`, which is the name of a field in
  the `result` dictionary passed to `scheduler.on_trial_report`. This
  field contains the resource $r$ for which metric values have been
  reported. For example, if a trial reports validation error at the end of
  the 5-th epoch of training, `result` contains `{resource_attr: 5}`.
* We already noted the arguments `max_resource_attr` and `max_t` in
  [FIFOScheduler](random_search.md#fifoscheduler-and-randomsearcher).
  They are used to determine the maximum resource $r_{max}$ (e.g., the
  total number of epochs a trial is to be trained, if not stopped before).
  As discussed in detail [here](../multifidelity/mf_setup.md#the-launcher-script),
  it is best practice reserving a field in the configuration space
  `scheduler.config_space` to contain $r_{max}$. If this is done, its name
  should be passed in `max_resource_attr`. Now, every
  configuration sent to the training script contains $r_{max}$, which
  should not be hardcoded in the script. Moreover, if `max_resource_attr`
  is used, a pause-and-resume scheduler (e.g., `HyperbandScheduler` with
  `type="stopping"`) can modify this field in the configuration of a trial
  which is only to be run until a certain resource less than $r_{max}$.
  Nevertheless, if `max_resource_attr` is not used, then $r_{max}$ has to
  be passed explicitly via `max_t` (which is not needed if
  `max_resource_attr` is used).
* `reduction_factor`, `grace_period`, `brackets` are important parameters
  detailed in the [tutorial](../multifidelity/README.md). If `brackets>1`, we
  run asynchronous Hyperband with this number of brackets, while for `bracket=1`
  we run asynchronous successive halving (this is the default).
* As detailed in the
  [tutorial](../multifidelity/mf_asha.md#asynchronous-successive-halving-early-stopping-variant),
  `type` determines whether the method uses early stopping (`type="stopping"`)
  or pause-and-resume scheduling (`type="promotion"`). Further choices of
  `type` activate specific algorithms such as RUSH, PASHA, or cost-sensitive
  successive halving.


## Kernel Density Estimator Searcher

One of the most flexible ways of extending `HyperbandScheduler` is to provide
it with a novel [searcher](first_example.md#searchers-and-schedulers). In order
to understand how this is done, we will walk through
[MultiFidelityKernelDensityEstimator](../../../syne_tune/optimizer/schedulers/searchers/kde/multi_fidelity_kde_searcher.py#L26)
and [KernelDensityEstimator](../../../syne_tune/optimizer/schedulers/searchers/kde/kde_searcher.py#L30).
This searcher implements `suggest` as in [BOHB](https://arxiv.org/abs/1807.01774),
as also detailed in the
[tutorial](../multifidelity/mf_sync_model.md#synchronous-bohb). In a nutshell,
the searcher splits all observations into two parts ("good" and "bad"), depending
on metric values lying above or below a certain quantile, and fits kernel density
estimators to these two subsets. It then makes decisions based on a particular
ratio of these densities, which is approximating a variant of the expected
improvement acquisition function.

We begin with the base class `KernelDensityEstimator`, which works together with
[FIFOScheduler](random_search.md#fifoscheduler-and-randomsearcher), but already
implements most of what is needed in the multi-fidelity context.
* The code does quite some bookkeeping concerned with mapping configurations
  to feature vectors. If you want to do this from scratch for your searcher,
  we recommend to use
  [HyperparameterRanges](../../../syne_tune/optimizer/schedulers/searchers/utils/hp_ranges.py#L36).
  However, `KernelDensityEstimator` was extracted from the original BOHB
  implementation.
* Observation data is collected in `self.X` (feature vectors for configurations)
  and `self.y` (values for `self._metric`, negated if `self.mode == "max"`). In
  particular, the `_update` method simply appends new data to these members.
* `get_config` fits KDEs to the "good" and "bad" parts of `self.X`, `self.y`.
  It then samples `self.num_candidates` configurations at random, evaluates the
  TPE acquisition function for each candidate, and returns the best one.

The class `MultiFidelityKernelDensityEstimator` inherits from
`KernelDensityEstimator`:
* On top of `self.X` and `self.y`, it also maintains resource values $r$ for
  each datapoint in `self.resource_levels`.
* `get_config` remains the same, only its subroutine `train_kde` for training
  the "good" and "bad" density models is modified. The idea is to fit these to
  data from a single rung level, namely the largest level at which we have
  observed at least `self.num_min_data_points` points.
* `configure_scheduler` restricts usage to `HyperbandScheduler` (asynchronous
  Hyperband) and `SynchronousHyperbandScheduler` (synchronous Hyperband).
  Also, `self.resource_attr` is obtained from the scheduler, so does not have
  to be passed.

While being functional and simple, the `MultiFidelityKernelDensityEstimator`
does not showcase the full range of information exchanged between
`HyperbandScheduler` and a searcher. In particular:
* `register_pending`: BOHB does not take pending evaluations into account.
* `remove_case`, `evaluation_failed` are not implemented.
* `get_state`, `clone_from_state` are not implemented, so schedulers with this
  searcher are not properly serialized.

For a more complete and advanced example, the reader is invited to study
[GPMultiFidelitySearcher](../../../syne_tune/optimizer/schedulers/searchers/gp_multifidelity_searcher.py#L43)
and [GPFIFOSearcher](../../../syne_tune/optimizer/schedulers/searchers/gp_fifo_searcher.py#L440)
This searcher takes pending evaluations into account (by way of fantasizing).
Moreover, it can be configured with a Gaussian process model and an acquisition
function, which is optimized in a gradient-based manner.

Moreover, as already noted [here](first_example.md#searchers-and-schedulers),
`HyperbandScheduler` also allows to configure the decision rule for stop/continue
or pause/resume as part of `on_trial_report`. Examples for this are found in
[StoppingRungSystem](../../../syne_tune/optimizer/schedulers/hyperband_stopping.py#L146),
[PromotionRungSystem](../../../syne_tune/optimizer/schedulers/hyperband_promotion.py#L21),
[RUSHStoppingRungSystem](../../../syne_tune/optimizer/schedulers/hyperband_rush.py#L80),
[PASHARungSystem](../../../syne_tune/optimizer/schedulers/hyperband_pasha.py#L17),
[CostPromotionRungSystem](../../../syne_tune/optimizer/schedulers/hyperband_cost_promotion.py#L20).


In the [next section](extend_sync_hb.md), we show how extensions of synchronous
successive halving and Hyperband can be implemented.
