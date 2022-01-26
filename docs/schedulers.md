# Syne Tune: Using the Built-in Schedulers

In this tutorial, you will learn how to use and configure the built-in HPO
algorithms. Alternatively, you can also use most algorithms from
[Ray Tune](https://docs.ray.io/en/master/tune/index.html).

First, make sure you have installed the `gpsearchers` and `benchmarks`
dependencies:

```bash
pip install -e .[gpsearchers,benchmarks]
```


## Schedulers and Searchers

The decision-making algorithms driving an HPO experiments are referred to as
**schedulers**. As in Ray Tune, some of our schedulers are internally
configured by a **searcher**. A scheduler interacts with the back-end, making
decisions on which configuration to evaluate next, and whether to stop, pause
or resume existing trials. It relays "next configuration" decisions to the
searcher. Some searchers maintain a **surrogate model** which is fitted to
metric data coming from evaluations.


## FIFOScheduler

This is the simplest kind of scheduler. It cannot stop or pause trials, each
evaluation proceeds to the end. Depending on the searcher, this scheduler
supports:

* Random search [`searcher=random`]
* Bayesian optimization with Gaussian processes [`searcher=bayesopt`]

Here is a launcher script using `FIFOScheduler`:
```python
import logging

from syne_tune.backend.local_backend import LocalBackend
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.tuner import Tuner
from syne_tune.stopping_criterion import StoppingCriterion

from examples.training_scripts.mlp_on_fashion_mnist.mlp_on_fashion_mnist import \
    mlp_fashionmnist_benchmark


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    n_workers = 4

    # We pick the MLP on FashionMNIST benchmark
    # The 'benchmark' dict contains arguments needed by scheduler and
    # searcher (e.g., 'mode', 'metric'), along with suggested default values
    # for other arguments (which you are free to override)
    benchmark = mlp_fashionmnist_benchmark({'dataset_path': './'})
    config_space = benchmark['config_space']

    backend = LocalBackend(entry_point=benchmark['script'])

    # GP-based Bayesian optimization searcher. Many options can be specified
    # via `search_options`, but let's use the defaults
    searcher = 'bayesopt'
    search_options = {'num_init_random': n_workers + 2}
    # FIFOScheduler. Together with searcher `bayesopt`, this selects Bayesian
    # optimization without early stopping.
    scheduler = FIFOScheduler(
        config_space,
        searcher=searcher,
        search_options=search_options,
        mode=benchmark['mode'],
        metric=benchmark['metric'])

    tuner = Tuner(
        backend=backend,
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_wallclock_time=120),
        n_workers=n_workers)

    tuner.run()
```

What happens in this launcher script?

* We select the `mlp_fashionmnist` benchmark, adopting its default hyperparameter
  search space without modifications.
* We select the local back-end, which runs up to `n_workers = 4` processes in
  parallel on the same instance.
* We create a `FIFOScheduler` with `searcher = 'bayesopt'`. This means that new
  configurations to be evaluated are selected by Bayesian optimization, and all
  trials are run to the end. The scheduler needs to know the `config_space`,
  the name of metric to tune (`metric`) and whether to minimize or maximize
  this metric (`mode`). For `mlp_fashionmnist`, we have `metric = 'accuracy'`
  and `mode = 'max'`, so we select a configuration which maximizes accuracy.
* Options for the searcher can be passed via `search_options`. We use defaults,
  instead of changing `num_init_random` (see below) to the number of workers
  plus two.
* Finally, we create the tuner, passing `backend`, `scheduler`, as well as the
  stopping criterion for the experiment (stop after 120 seconds) and the number
  of workers. The experiment is started by `tuner.run()`.

The full range of arguments of `FIFOScheduler` is documented in
[syne_tune/optimizer/schedulers/fifo.py](../syne_tune/optimizer/schedulers/fifo.py).
Here, we list the most important ones:

* `config_space`: Hyperparameter search space. This argument is mandatory. Apart
  from hyperparameters to be searched over, the space may contain fixed
  parameters (such as `epochs` in the example above). A `config` passed to
  the training script is always extended by these fixed parameters.
  If you use a benchmark, you can use `benchmark['config_space']` here, or
  you can modify this default search space.
* `searcher`: Selects searcher to be used (see below).
* `search_options`: Options to configure the searcher (see below).
* `metric`, `mode`: Name of metric to tune (i.e, key used in `report` call by
  the training script), which is either to be minimized (`mode = 'min'`) or
  maximized (`mode = 'max'`). If you use a benchmark, just use
  `benchmark['metric']` and `benchmark['mode']` here.
* `points_to_evaluate`: Allows to specify a list of configurations which are
  evaluated first. If your training code corresponds to some open source ML
  algorithm, you may want to use the defaults provided in the code. The
  entry (or entries) in `points_to_evaluate` do not have to specify values
  for all hyperparameters. For any hyperparameter not listed there, the
  following rule is used to choose a default. For `float` and `int` value
  type, the mid-point of the search range is used (in linear or log
  scaling). For categorical value type, the first entry in the value set
  is used.
  The default is a single config with all values chosen by the default rule.
  Pass an empty list in order to not specify any initial configs.
* `random_seed`: Master random seed. Random sampling in schedulers and searchers
  are done by a number of `numpy.random.RandomState` generators, whose seeds
  are derived from `random_seed`. If not given, a random seed is sampled.

### Random Search (searcher = 'random')

The simplest HPO baseline is **random search**, which you obtain with
`searcher='random'`. Search decisions are not based on past data, a new
configuration is chosen by sampling attribute values at random, from
distributions specified in `config_space`:

* `search_space.uniform(lower, upper)`: Real-valued uniform in `[lower, upper]`
* `search_space.loguniform(lower, upper)`: Real-valued log-uniform in
  `[lower, upper]`. More precisely, the value is `exp(x)`, where `x` is drawn
  uniformly in `[log(lower), log(upper)]`
* `search_space.randint(lower, upper)`: Integer uniform in `lower, ..., upper`.
  The value range includes both `lower` and `upper` (difference to Python range
  convention)
* `search_space.lograndint(lower, upper)`: Integer log-uniform in
  `lower, ..., upper`. More precisely, the value is `int(round(exp(x)))`, where
  `x` is drawn uniformly in `[log(lower - 0.5), log(upper + 0.5)]`
* `search_space.choice(categories)`: Uniform from the finite list `categories`
  of `str` values

If `points_to_evaluate` is specified, configurations are first taken from this
list before any are drawn at random. Options for configuring the searcher are
given in `search_options`. These are:

* `debug_log`: If `True` (default), a useful log output about the search progress
  is printed.

### Bayesian Optimization (searcher = 'bayesopt')

**Bayesian optimization** is obtained by `searcher='bayesopt'`. A good overview
of Bayesian optimization for HPO is provided in
[Practical Bayesian Optimization of Machine Learning Algorithms](https://papers.nips.cc/paper/2012/file/05311655a15b75fab86956663e1819cd-Paper.pdf):
```bibtex
@article{
   title={Practical {Bayesian} Optimization of Machine Learning Algorithms},
   author={Snoek, J. and Larochelle, H. and Adams, R.},
   booktitle={Neural Information Processing Systems 25},
   year={2012},
   pages={2951--2959}
}
```
Options for configuring the searcher are given in `search_options`. These
include options for the random searcher. The full range of arguments of
`GPFIFOSearcher` is documented in
[syne_tune/optimizer/schedulers/searchers/gp_fifo_searcher.py](../syne_tune/optimizer/schedulers/searchers/gp_fifo_searcher.py).
Here, we list the most important ones:

* `num_init_random`: Number of initial configurations chosen at random (or via
  `points_to_evaluate`). In fact, the number of initial configurations is the
  maximum of this and the length of `points_to_evaluate`. Afterwards,
  configurations are chosen by Bayesian optimization (BO). In general, BO is
  only used once at least one metric value from past trials is available. We
  recommend to set this value to the number of workers plus two.
* `opt_nstarts`, `opt_maxiter`: BO employs a Gaussian process surrogate model,
  whose own hyperparameters (e.g., kernel parameters, noise variance) are
  chosen by empirical Bayesian optimization. In general, this is done whenever
  new data becomes available. It is the most expensive computation in each
  round. `opt_maxiter` is the maximum number of L-BFGS iterations. We run
  `opt_nstarts` such optimizations from random starting points and pick the
  best.
* `opt_skip_init_length`, `opt_skip_period`: Refitting the GP hyperparameters in
  each round can become expensive, especially when the number of observations
  grows large. If so, you can choose to do it only every `opt_skip_period`
  rounds. Skipping optimizations is done only once the number of observations
  is above `opt_skip_init_length`.
* `map_reward`: Internally, the criterion is minimized. If `mode='max'` for
  your tuning function (so you maximize a reward), you can specify how this
  reward is mapped to the inner criterion. Choices are 'minus_x'
  (`criterion = -reward`) and '{a}_minus_x', where {a} is a constant
  `(criterion = {a} - reward`. For example, '1_minus_x' maps accuracy to error.


## HyperbandScheduler

This scheduler comes in two different variants, one may stop trials early,
the other may pause trials and resume them later. For tuning
neural network models, it tends to work much better than `FIFOScheduler`. You
may have read about successive halving and Hyperband before. Chances are you
read about **synchronous scheduling** of parallel evaluations, while both
`HyperbandScheduler` and `FIFOScheduler` implement **asynchronous scheduling**,
which is different. The papers cited below provide a detailed overview of
asynchronous variants of successive halving, and of the algorithms discussed
here. Experiments therein indicate that asynchronous scheduling can be far more
efficient for HPO than synchronous scheduling. At present, Syne Tune supports
synchronous random search (by passing the argument `asynchronous_scheduling=False`
when creating the `Tuner` object), but does not yet support synchronous
Hyperband.

Hyperband is an extension of successive halving to multiple brackets. We will
discuss successive halving, mentioning Hyperband later. In our experience so
far, asynchronous successive halving does not profit from multiple brackets if
applied to neural network tuning.

Here is a launcher script using `HyperbandScheduler`:
```python
import logging

from syne_tune.backend.local_backend import LocalBackend
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.tuner import Tuner
from syne_tune.stopping_criterion import StoppingCriterion

from examples.training_scripts.mlp_on_fashion_mnist.mlp_on_fashion_mnist import \
    mlp_fashionmnist_benchmark


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    n_workers = 4

    # We pick the MLP on FashionMNIST benchmark
    # The 'benchmark' dict contains arguments needed by scheduler and
    # searcher (e.g., 'mode', 'metric'), along with suggested default values
    # for other arguments (which you are free to override)
    benchmark = mlp_fashionmnist_benchmark({'dataset_path': './'})
    config_space = benchmark['config_space']

    backend = LocalBackend(entry_point=benchmark['script'])

    # GP-based Bayesian optimization searcher. Many options can be specified
    # via `search_options`, but let's use the defaults
    searcher = 'bayesopt'
    search_options = {'num_init_random': n_workers + 2}
    # Hyperband (or successive halving) scheduler of the stopping type.
    # Together with 'bayesopt', this selects the MOBSTER algorithm.
    default_params = benchmark['default_params']
    scheduler = HyperbandScheduler(
        config_space,
        searcher=searcher,
        search_options=search_options,
        type='stopping',
        max_t=default_params['epochs'],
        grace_period=default_params['grace_period'],
        reduction_factor=default_params['reduction_factor'],
        resource_attr=benchmark['resource_attr'],
        mode=benchmark['mode'],
        metric=benchmark['metric'])

    tuner = Tuner(
        backend=backend,
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_wallclock_time=120),
        n_workers=n_workers,
    )

    tuner.run()
```

Much of this launcher script is the same as for `FIFOScheduler`, but
`HyperbandScheduler` comes with a number of extra arguments we will explain in
the sequel (`type`, `max_t`, `grace_period`, `reduction_factor`,
`resource_attr`). The `mlp_fashionmnist` benchmark trains a two-layer MLP on
`FashionMNIST` (see
[mlp_on_fashion_mnist.py](../benchmarking/training_scripts/mlp_on_fashion_mnist/mlp_on_fashion_mnist.py)).
The accuracy is computed and reported at the end of each epoch:

```python
for epoch in range(resume_from + 1, config['epochs'] + 1):
    train_model(config, state, train_loader)
    accuracy = validate_model(config, state, valid_loader)
    report(
        epoch=epoch,
        accuracy=accuracy)
```

While `metric = 'accuracy'` is the criterion to be optimized,
`resource_attr = 'epoch'` is the resource attribute. In the schedulers
discussed here, the resource attribute must be a positive integer.

`HyperbandScheduler` maintains reported metrics for all trials at certain
**rung levels** (levels of resource attribute `epoch` at which scheduling
decisions are done). When a trial reports `(epoch, accuracy)` for a rung level
`== epoch`, the scheduler makes a decision whether to stop (pause) or continue.
This decision is done based on all `accuracy` values encountered before
at the same rung level. Whenever a trial is stopped (or paused), the executing
worker becomes available to evaluate a different configuration.

Rung level spacing and stop/go decisions are determined by the parameters
`max_t`, `grace_period`, and `reduction_factor`. Rung levels are `grace_period,
grace_period * eta, grace_period * (eta ** 2), ..., max_t`, where
`eta = reduction_factor`. In the example above, `max_t = 81`, `grace_period = 1`,
and `reduction_factor = 3`, so that rung levels are 1, 3, 9, 27, 81. The spacing
is such that stop/go decisions are done less frequently for trials which already
went further: they have earned trust by not being stopped earlier. `max_t` need
not be of the form `grace_period * (eta ** k)`. If `max_t = 56` in the example
above, the rung levels would be 1, 3, 9, 27, 56.

If `max_t` is not given as argument to `HyperbandScheduler`, the value may be
inferred from `config_space`. Namely, `config_space['epochs']`,
`config_space['max-t']`, `config_space['max-epochs']` are checked in this order.
In the example above, `configspace['epochs']` contains the correct value, so we
could have dropped `max_t`.

Given such a rung level spacing, stop/go decisions are done by comparing
`accuracy` to the `1 / reduction_factor` quantile of values recorded at the
rung level. In the example above, our trial is stopped if `accuracy` is no
better than the best 1/3 of previous values (the list includes the current
`accuracy` value), otherwise it is stopped.

As detailed in
[Model-based Asynchronous Hyperparameter and Neural Architecture Search](https://arxiv.org/abs/2003.10865),
there are two different types of asynchronous successive halving, selected by
the `type` argument:

* **Stopping-based asynchronous successive halving** [`type='stopping'`]:
  This is essentially a refined variant of early stopping for HPO. If a
  stop/go decision comes out 'stop', the trial is terminated, otherwise it
  may continue. If there are less than `reduction_factor` recorded values
  at the rung level (including the current one), the trial continues.
  Moreover, whenever a worker is free, a trial is started with a newly
  chosen configuration. This variant is simple and does not require the
  back-end to pause and resume trials. It is the default for `type`.
* **Promotion-based asynchronous successive halving** [`type = 'promotion'`]:
  This variant has been proposed as **ASHA** in
  [A System for Massively Parallel Hyperparameter Tuning](https://arxiv.org/abs/1810.05934).
  If a stop/go decision comes out 'stop', the trial is paused, otherwise it
  may continue. If there are less than `reduction_factor` recorded values
  at the rung level (including the current one), the trial is paused.
  Moreover, whenever a worker is free, the scheduler first scans all paused
  trials in reverse rung level order (largest rung levels first). If for any
  of them, the stop/go decision comes out 'go', this trial is resumed.
  Otherwise, if none of the paused trials can be resumed, a trial is started with
  a newly chosen configuration. This variant requires the back-end to pause
  and resume trials (which typically includes support for checkpointing).
* **Progressive ASHA (PASHA)** [`type='pasha'`]:
  This is the variant of ASHA presented in [TODO:ADDLINK](htpp://#).
  This variant of ASHA have been developed to be resource-efficient on large datasets
  and it works by progressively extending the maximum resources level at which configurations are trained. 
  It is empirically shown that it is often possible to identify optimal configurations
  early on, but it is often difficult to determine how reliable an early decision is.
  PASHA tries to automatically identify the minimum resources level at which it can perform
  a reliable decision. It can be used in situations where re-training a model several
  time would be just too expensive.

The full range of arguments of `HyperbandScheduler` is documented in
[syne_tune/optimizer/schedulers/hyperband.py](../syne_tune/optimizer/schedulers/hyperband.py).
It includes all those of `FIFOScheduler`. Here, we list the most important ones:

* `max_t`, `grace_period`, `reduction_factor`: As detailed above, these determine
  the rung levels and the stop/go decisions. The resource attribute is a
  positive integer. We need `reduction_factor >= 2`.
* `rung_levels`: Alternatively, the user can specify the list of rung levels
  directly (positive integers, strictly increasing). The stop/go rule in
  the successive halving scheduler is set based on the ratio of successive rung
  levels.
* `type`: Values are `'stopping', 'promotion'` (see above).
* `brackets`: Number of brackets to be used in Hyperband. The default is 1,
  which corresponds to successive halving. Each bracket has a different
  `grace_period`, they share `max_t` and `reduction_factor`. When starting a
   new trial, it is assigned a randomly sampled bracket (smaller brackets have
   a higher probability). The larger the bracket, the larger `grace_period`
   before the trial has to compete with others.
* `rung_system_per_bracket`: Only used if `brackets > 1`. If `True`, each
   bracket maintains its own rung level system, so that trials only compete
   with those started in the same bracket. If `False`, all trials compete
   with each other in a single rung level system, they just get different
   head starts in terms of their `grace_period`.
* `searcher_data`: This option is relevant when `searcher='bayesopt'` and
   is discussed below.

### Asynchronous Hyperband ASHA (searcher = 'random')

If `HyperbandScheduler` is configured with a random searcher, we obtain ASHA,
as proposed in [A System for Massively Parallel Hyperparameter Tuning](https://arxiv.org/abs/1810.05934).
```bibtex
@article{
   title={A System for Massively Parallel Hyperparameter Tuning},
   author={Liam Li and Kevin Jamieson and Afshin Rostamizadeh and Ekaterina Gonina and Moritz Hardt and Benjamin Recht and Ameet Talwalkar},
   journal={arXiv preprint arXiv:1810.05934}
}
```

Strictly speaking, their paper details the promotion-based variant
(`type = 'promotion'`), while the stopping-based variant is based on earlier
ideas like the **median rule**.

Nothing much can be configured via `search_options` in this case. The arguments
are the same as for random search with `FIFOScheduler`.

### Model-based Asynchronous Hyperband MOBSTER (searcher = 'bayesopt')

If `HyperbandScheduler` is configured with a Bayesian optimization searcher, we
obtain MOBSTER, as proposed in
[Model-based Asynchronous Hyperparameter and Neural Architecture Search](https://arxiv.org/abs/2003.10865).
```bibtex
@article{
   title={Model-based Asynchronous Hyperparameter and Neural Architecture Search},
   author={Aaron Klein and Louis C. Tiao and Thibaut Lienart and Cedric Archambeau and Matthias Seeger},
   journal={arXiv preprint arXiv:2003.10865}
}
```

MOBSTER uses a multi-task Gaussian process surrogate model for metrics data
observed at all resource levels. Options for configuring the searcher are given
in `search_options`. These include options for the random searcher. The full
range of arguments of `GPMultiFidelitySearcher` is documented in
[syne_tune/optimizer/schedulers/searchers/gp_multifidelity_searcher.py](../syne_tune/optimizer/schedulers/searchers/gp_multifidelity_searcher.py).
Here, we list the most important ones:

* `num_init_random`: See `FIFOSearcher`, `searcher='bayesopt'`.
* `opt_nstarts`, `opt_maxiter`: See `FIFOSearcher`, `searcher='bayesopt'`.
* `opt_skip_init_length`, `opt_skip_period`: See `FIFOSearcher`,
  `searcher='bayesopt'`.
* `map_reward`: See `FIFOSearcher`, `searcher='bayesopt'`.
* `gp_resource_kernel`: Values are `'matern52', 'matern52-res-warp',
   'exp-decay-sum', 'exp-decay-delta1', 'exp-decay-combined'`. Selects different
   multi-task GP surrogate models. For details, please see the code. The
   default choice is `'exp-decay-sum'`, which is closely related to the kernel
   proposed in[Freeze-Thaw Bayesian Optimization](https://arxiv.org/abs/1406.3896),
   but without the conditional independence assumptions made there.
*  `opt_skip_num_max_resource`: Alternative to `opt_skip_period`. If `True`,
   the GP surrogate model hyperparameters are refit only when a trial reaches
   level `max_t`.
*  `resource_acq_bohb_threshold`: MOBSTER is choosing a new configuration by
   maximizing the expected improvement (EI) acquisition function at a certain
   resource level `r_acq`. Since we are ultimately interested in performance
   at `r = max_t`, we would like to set `r_acq = max_t` as early as possible.
   On the other hand, EI may not be reliable at a resource level if too little
   metric data has been observed there (i.e., too few trials reached this
   level). MOBSTER is setting `r_acq` to the largest rung level `r <= max_t`
   for which at least `resource_acq_bohb_threshold` metric values have been
   recorded.

Finally, for `searcher='bayesopt'`, the `HyperbandScheduler` argument
`searcher_data` is relevant. Values are `'rungs', 'all', 'rungs_and_last'`.
Recall that the searcher represents past data by a multi-task GP surrogate
model conditioned on observed metric data. Inference in such a model scales
cubically in the number of datapoints. `searcher_data` determines which
metric observations are passed to the searcher to update its surrogate model:

* `'all'`: All observations at all resource levels are used for the surrogate
   model. This provides the best fit, but can be expensive, which can slow
   down the search.
* `'rungs'`: This is the default. Observations are used for the surrogate model
   only if their resource level is equal to a rung level. This renders the
   surrogate model cheaper, but may result in a worse fit.
* `'rungs_and_last'`: Observations are used for the surrogate model if their
   resource level is equal to a rung level, or if they are the last recent
   observation of a trial. The surrogate model is only a bit more expensive,
   but all most recent observations are used.


## Recommendations

Finally, we provide some general recommendations on how to use our built-in
schedulers.

* If you can afford it for your problem, random search (`FIFOScheduler`,
  `searcher='random'`) is a useful baseline. However, if even a single
  full evaluation takes a long time, try ASHA instead (`HyperbandScheduler`,
  `searcher='random'`, `type='stopping'` or `type='promotion'`).
* Use these baseline runs to get an idea how long your experiment needs
  to run. It is recommended to use a stopping criterion of the form
  `stop_criterion=StoppingCriterion(max_wallclock_time=X)`, so that the
  experiment is stopped after `X` seconds.
* If your tuning problem comes with an obvious resource parameter, make sure
  to implement it such that results are reported during the evaluation, not
  only at the end. When training a neural network model, choose the number
  of epochs as resource. In other situations, choosing a resource parameter may be
  more difficult. Our schedulers require positive (or non-negative) integers.
  Make sure that evaluations for the same configuration scale linearly in
  the resource parameter: an evaluation up to `2 * r` should be roughly
  twice as expensive as one up to `r`.
* If your problem has a resource parameter, always make sure to try
  `HyperbandScheduler`, which in many cases runs much faster than
  `FIFOScheduler`.
* If you end up tuning the same ML algorithm or neural network model on
  different datasets, make sure to set `points_to_evaluate` appropriately. If
  the model comes from frequently used open source code, its built-in
  defaults will be a good choice. Any hyperparameter not covered in
  `points_to_evaluate` is set using a "midpoint" heuristic. While still better
  than choosing the first configuration at random, this may not be very good.
* For `HyperbandScheduler`, you need to choose between `type='stopping'` and
  `type='promotion'`. For neural network tuning, start with `'stopping'`,
  which is simpler and does not need checkpointing. However, if checkpointing
  is in place, try both of them. For some problems, the
  notion of stopping and checkpointing does not apply, and `'promotion'` may
  be more natural. For example, you may train your model on subsamples of size
  `(r / 10) * total_size`, `r=1,...,10` (assuming that training scales roughly
  linear in the dataset size). Training for every `r` has to start from
  scratch in this case.
* In general, the defaults should work well if your tuning problem is
  expensive enough (at least a few minutes per unit of `r`). In such cases,
  MOBSTER (`HyperbandScheduler`, `searcher='bayesopt'`) can outperform ASHA
  substantially. However, if your problem is cheap, so you can afford a lot
  of evaluation, the searchers based on GP surrogate models may end up
  expensive. With ASHA your baseline, you can try to speed up MOBSTER by
  changing `opt_skip_period` (or using `opt_skip_num_max_resource`).
