Using the Built-in Schedulers
=============================

In this tutorial, you will learn how to use and configure the most important
built-in HPO algorithms. Alternatively, you can also use most algorithms from
`Ray Tune <https://docs.ray.io/en/master/tune/index.html>`__.

`This tutorial <tutorials/basics/README.html>`__ provides a walkthrough of
some of the topics addressed here.

Schedulers and Searchers
------------------------

The decision-making algorithms driving an HPO experiments are referred to as
**schedulers**. As in Ray Tune, some of our schedulers are internally configured
by a **searcher**. A scheduler interacts with the backend, making decisions on
which configuration to evaluate next, and whether to stop, pause or resume
existing trials. It relays “next configuration” decisions to the searcher. Some
searchers maintain a **surrogate model** which is fitted to metric data coming
from evaluations.

.. note::
   There are two ways to create many of the schedulers of Syne Tune:

   * Import wrapper class from :mod:`syne_tune.optimizer.baselines`, for example
     :class:`~syne_tune.optimizer.baselines.RandomSearch` for random search
   * Use template classes :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
     or :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` together with
     the ``searcher`` argument, for example ``FIFOScheduler`` with
     ``searcher="random"`` for random search

   Importing from :mod:`syne_tune.optimizer.baselines` is often simpler. However,
   in this tutorial, we will use the template classes in order to expose the
   common structure and to explain arguments only once.

FIFOScheduler
-------------

This is the simplest kind of scheduler. It cannot stop or pause trials, each
evaluation proceeds to the end. Depending on the searcher, this scheduler
supports:

* Random search [``searcher="random"``]
* Bayesian optimization with Gaussian processes [``searcher="bayesopt"``]
* Grid search [``searcher="grid"``]
* TPE with kernel density estimators [``searcher="kde"``]
* Constrained Bayesian optimization [``searcher="bayesopt_constrained"``]
* Cost-aware Bayesian optimization [``searcher="bayesopt_cost"``]
* Bore [``searcher="bore"``]

We will only consider the first two searchers in this tutorial. Here is a
launcher script using :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`:

.. code-block:: python

   import logging

   from syne_tune.backend import LocalBackend
   from syne_tune.optimizer.schedulers import FIFOScheduler
   from syne_tune import Tuner, StoppingCriterion

   from benchmarking.commons.benchmark_definitions.mlp_on_fashionmnist import \
       mlp_fashionmnist_benchmark


   if __name__ == '__main__':
       logging.getLogger().setLevel(logging.DEBUG)
       n_workers = 4
       max_wallclock_time = 120

       # We pick the MLP on FashionMNIST benchmark
       # The 'benchmark' object contains arguments needed by scheduler and
       # searcher (e.g., 'mode', 'metric'), along with suggested default values
       # for other arguments (which you are free to override)
       benchmark = mlp_fashionmnist_benchmark()
       config_space = benchmark.config_space

       backend = LocalBackend(entry_point=benchmark.script)

       # GP-based Bayesian optimization searcher. Many options can be specified
       # via `search_options`, but let's use the defaults
       searcher = "bayesopt"
       search_options = {'num_init_random': n_workers + 2}
       scheduler = FIFOScheduler(
           config_space,
           searcher=searcher,
           search_options=search_options,
           mode=benchmark.mode,
           metric=benchmark.metric,
       )

       tuner = Tuner(
           trial_backend=backend,
           scheduler=scheduler,
           stop_criterion=StoppingCriterion(
               max_wallclock_time=max_wallclock_time
           ),
           n_workers=n_workers,
       )

       tuner.run()

What happens in this launcher script?

* We select the ``mlp_fashionmnist`` benchmark, adopting its default
  hyperparameter search space without modifications.
* We select the local backend, which runs up to ``n_workers = 4`` processes in
  parallel on the same instance.
* We create a ``FIFOScheduler`` with ``searcher = "bayesopt"``. This means that
  new configurations to be evaluated are selected by Bayesian optimization, and
  all trials are run to the end. The scheduler needs to know the
  ``config_space``, the name of metric to tune (``metric``) and whether to
  minimize or maximize this metric (``mode``). For ``mlp_fashionmnist``, we
  have ``metric = "accuracy"`` and ``mode = "max"``, so we select a
  configuration which maximizes accuracy.
* Options for the searcher can be passed via ``search_options``. We use
  defaults, except for changing ``num_init_random`` (see below) to the number
  of workers plus two.
* Finally, we create the tuner, passing ``trial_backend``, ``scheduler``, as
  well as the stopping criterion for the experiment (stop after 120 seconds)
  and the number of workers. The experiment is started by ``tuner.run()``.

:class:`~syne_tune.optimizer.schedulers.FIFOScheduler` provides the full range
of arguments. Here, we list the most important ones:

* ``config_space``: Hyperparameter search space. This argument is mandatory.
  Apart from hyperparameters to be searched over, the space may contain fixed
  parameters (such as ``epochs`` in the example above). A ``config`` passed to
  the training script is always extended by these fixed parameters. If you use
  a benchmark, you can use ``benchmark["config_space"]`` here, or you can
  modify this default search space.
* ``searcher``: Selects searcher to be used (see below).
* ``search_options``: Options to configure the searcher (see below).
* ``metric``, ``mode``: Name of metric to tune (i.e, key used in ``report``
  call by the training script), which is either to be minimized (``mode="min"``)
  or maximized (``mode="max"``). If you use a benchmark, just use
  ``benchmark["metric"]`` and ``benchmark["mode"]`` here.
* ``points_to_evaluate``: Allows to specify a list of configurations which are
  evaluated first. If your training code corresponds to some open source ML
  algorithm, you may want to use the defaults provided in the code. The entry
  (or entries) in ``points_to_evaluate`` do not have to specify values for all
  hyperparameters. For any hyperparameter not listed there, the following rule
  is used to choose a default. For ``float`` and ``int`` value type, the
  mid-point of the search range is used (in linear or log scaling). For
  categorical value type, the first entry in the value set is used. The default
  is a single config with all values chosen by the default rule. Pass an empty
  list in order to not specify any initial configs.
* ``random_seed``: Master random seed. Random sampling in schedulers and
  searchers are done by a number of ``numpy.random.RandomState`` generators,
  whose seeds are derived from ``random_seed``. If not given, a random seed is
  sampled and printed in the log.

Random Search
~~~~~~~~~~~~~

The simplest HPO baseline is **random search**, which you obtain with
``searcher="random"``, or by using
:class:`~syne_tune.optimizer.baselines.RandomSearch` instead of
``FIFOScheduler``. Search decisions are not based on past data, a new
configuration is chosen by sampling attribute values at random, from
distributions specified in ``config_space``. These distributions are detailed
`here <search_space.html#domains>`__.

If ``points_to_evaluate`` is specified, configurations are first taken from
this list before any are drawn at random. Options for configuring the searcher
are given in ``search_options``. These are:

* ``debug_log``: If ``True``, a useful log output about the search progress is
  printed.
* ``allow_duplicates``: If ``True``, the same configuration may be suggested
  more than once. The default is ``False``, in that sampling is without
  replacement.

Bayesian Optimization
~~~~~~~~~~~~~~~~~~~~~

**Bayesian optimization** is obtained by ``searcher='bayesopt'``, or by using
:class:`~syne_tune.optimizer.baselines.BayesianOptimization` instead of
``FIFOScheduler``. More information about Bayesian optimization is provided
`here <tutorials/basics/basics_bayesopt.html>`__.

Options for configuring the searcher are given in ``search_options``. These
include options for the random searcher.
:class:`~syne_tune.optimizer.schedulers.searchers.GPFIFOSearcher` provides the
full range of arguments. We list the most important ones:

* ``num_init_random``: Number of initial configurations chosen at random (or
  via ``points_to_evaluate``). In fact, the number of initial configurations
  is the maximum of this and the length of ``points_to_evaluate``. Afterwards,
  configurations are chosen by Bayesian optimization (BO). In general, BO is
  only used once at least one metric value from past trials is available. We
  recommend to set this value to the number of workers plus two.
* ``opt_nstarts``, ``opt_maxiter``: BO employs a Gaussian process surrogate
  model, whose own hyperparameters (e.g., kernel parameters, noise variance)
  are chosen by empirical Bayesian optimization. In general, this is done
  whenever new data becomes available. It is the most expensive computation in
  each round. ``opt_maxiter`` is the maximum number of L-BFGS iterations. We
  run ``opt_nstarts`` such optimizations from random starting points and pick
  the best.
* ``max_size_data_for_model``, ``max_size_top_fraction``: GP computations scale
  cubically with the number of observations, and decision making can become
  very slow for too many trials. Whenever there are more than
  ``max_size_data_for_model`` observations, the dataset is downsampled to this
  size. Here, ``max_size_data_for_model * max_size_top_fraction`` of the entries
  correspond to the cases with the best metric values, while the remaining
  entries are drawn at random (without replacement) from all other cases.
  Defaults to
  :const:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.defaults.DEFAULT_MAX_SIZE_DATA_FOR_MODEL`.
* ``opt_skip_init_length``, ``opt_skip_period``: Refitting the GP
  hyperparameters in each round can become expensive, especially when the
  number of observations grows large. If so, you can choose to do it only
  every ``opt_skip_period`` rounds. Skipping optimizations is done only once
  the number of observations is above ``opt_skip_init_length``.
* ``gp_base_kernel``: Selects the covariance (or kernel) function to be used in
  the surrogate model. Current choices are "matern52-ard" (Matern ``5/2`` with
  automatic relevance determination; the default) and "matern52-noard"
  (Matern ``5/2`` without ARD).
* ``input_warping``: If this is ``True``, inputs are warped before being fed
  into the covariance function, the effective kernel becomes
  :math:`k(w(x), w(x'))`, where :math:`w(x)` is a warping transform with two
  non-negative parameters per component. These parameters are learned along with
  other parameters of the surrogate model. Input warping allows the surrogate
  model to represent non-stationary functions, while still keeping the numbers
  of parameters small. Note that only such components of :math:`x` are warped
  which belong to non-categorical hyperparameters.
* ``boxcox_transform``: If this is ``True``, target values are transformed before
  being fitted with a Gaussian marginal likelihood. This is using the Box-Cox
  transform with a parameter :math:`\lambda`, which is learned alongside other
  parameters of the surrogate model. The transform is :math:`\log y` for
  :math:`\lambda = 0`, and :math:`y - 1` for :math:`\lambda = 1`. This option
  requires the targets to be positive.

HyperbandScheduler
------------------

This scheduler comes in at least two different variants, one may stop trials
early (``type="stopping"``), the other may pause trials and resume them later
(``type="promotion"``). For tuning neural network models, it tends to work
much better than ``FIFOScheduler``. You may have read about successive halving
and Hyperband before. Chances are you read about **synchronous scheduling** of
parallel evaluations, while both ``HyperbandScheduler`` and ``FIFOScheduler``
implement **asynchronous scheduling**, which can be substantially more
efficient. `This tutorial <tutorials/multifidelity/README.html>`__ provides
details about synchronous and asynchronous variants of successive halving and
Hyperband.

Here is a launcher script using
:class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`:

.. code-block:: python

   import logging

   from syne_tune.backend import LocalBackend
   from syne_tune.optimizer.schedulers import HyperbandScheduler
   from syne_tune import Tuner, StoppingCriterion

   from benchmarking.commons.benchmark_definitions.mlp_on_fashionmnist import \
       mlp_fashionmnist_benchmark

   if __name__ == '__main__':
       logging.getLogger().setLevel(logging.DEBUG)
       n_workers = 4
       max_wallclock_time = 120

       # We pick the MLP on FashionMNIST benchmark
       # The 'benchmark' object contains arguments needed by scheduler and
       # searcher (e.g., 'mode', 'metric'), along with suggested default values
       # for other arguments (which you are free to override)
       benchmark = mlp_fashionmnist_benchmark()
       config_space = benchmark.config_space

       backend = LocalBackend(entry_point=benchmark.script)

       # MOBSTER: Combination of asynchronous successive halving with
       # GP-based Bayesian optimization
       searcher = 'bayesopt'
       search_options = {'num_init_random': n_workers + 2}
       scheduler = HyperbandScheduler(
           config_space,
           searcher=searcher,
           search_options=search_options,
           type="stopping",
           max_resource_attr=benchmark.max_resource_attr,
           resource_attr=benchmark.resource_attr,
           mode=benchmark.mode,
           metric=benchmark.metric,
           grace_period=1,
           reduction_factor=3,
       )

       tuner = Tuner(
           trial_backend=backend,
           scheduler=scheduler,
           stop_criterion=StoppingCriterion(
               max_wallclock_time=max_wallclock_time
           ),
           n_workers=n_workers,
       )

       tuner.run()

Much of this launcher script is the same as for ``FIFOScheduler``, but
:class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` comes with a number
of extra arguments we will explain in the sequel (``type``,
``max_resource_attr``, ``grace_period``, ``reduction_factor``,
``resource_attr``). The ``mlp_fashionmnist`` benchmark trains a two-layer MLP
on ``FashionMNIST`` (more details are
`here <tutorials/basics/basics_setup.html>`__). The accuracy is computed and
reported at the end of each epoch:

.. code-block:: python

   for epoch in range(resume_from + 1, config['epochs'] + 1):
       train_model(config, state, train_loader)
       accuracy = validate_model(config, state, valid_loader)
       report(epoch=epoch, accuracy=accuracy)

While ``metric="accuracy"`` is the criterion to be optimized,
``resource_attr="epoch"`` is the resource attribute. In the schedulers
discussed here, the resource attribute must be a positive integer.

:class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` maintains reported
metrics for all trials at certain **rung levels** (levels of resource attribute
``epoch`` at which scheduling decisions are done). When a trial reports
``(epoch, accuracy)`` for a rung level ``== epoch``, the scheduler makes a
decision whether to stop (pause) or continue. This decision is done based on
all ``accuracy`` values encountered before at the same rung level. Whenever a
trial is stopped (or paused), the executing worker becomes available to evaluate
a different configuration.

Rung level spacing and stop/go decisions are determined by the parameters
``max_resource_attr``, ``grace_period``, and ``reduction_factor``. The first
is the name of the attribute in ``config_space`` which contains the maximum
number of epochs to train (``max_resource_attr == "epochs"`` in our
benchmark). This allows the training script to obtain
``max_resource_value = config["max_resource_attr"]``. Rung levels are
:math:`r_{min}, r_{min} \eta, r_{min} \eta^2, \dots, r_{max}`, where
:math:`r_{min}` is ``grace_period``, :math:`\eta` is ``reduction_factor``, and
:math:`r_{max}` is ``max_resource_value``. In the example above,
``max_resource_value = 81``, ``grace_period = 1``, and ``reduction_factor = 3``,
so that rung levels are 1, 3, 9, 27, 81. The spacing is such that stop/go
decisions are done less frequently for trials which already went further: they
have earned trust by not being stopped earlier. :math:`r_{max}` need not be
of the form :math:`r_{min} \eta^k`. If ``max_resource_value = 56`` in the
example above, the rung levels would be 1, 3, 9, 27, 56.

Given such a rung level spacing, stop/go decisions are done by comparing
``accuracy`` to the ``1 / reduction_factor`` quantile of values recorded at
the rung level. In the example above, our trial is stopped if ``accuracy`` is
no better than the best 1/3 of previous values (the list includes the current
``accuracy`` value), otherwise it is stopped.

Further details about ``HyperbandScheduler`` and multi-fidelity HPO methods
are given in `this tutorial <tutorials/multifidelity/README.html>`__.
:class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` provides the full
range of arguments. Here, we list the most important ones:

* ``max_resource_attr``, ``grace_period``, ``reduction_factor``: As detailed
  above, these determine the rung levels and the stop/go decisions. The
  resource attribute is a positive integer. We need ``reduction_factor >= 2``.
  Note that instead of ``max_resource_attr``, you can also use ``max_t``,
  as detailed
  `here <tutorials/multifidelity/mf_setup.html#the-launcher-script>`__.
* ``rung_increment``: This parameter can be used instead of ``reduction_factor``
  (the latter takes precedence). In this case, rung levels are spaced linearly:
  :math:`r_{min} + j \nu, j = 0, 1, 2, \dots`, where :math:`\nu` is
  ``rung_increment``. The stop/go rule in the successive halving scheduler is
  set based on the ratio of successive rung levels.
* ``rung_levels``: Alternatively, the user can specify the list of rung levels
  directly (positive integers, strictly increasing). The stop/go rule in the
  successive halving scheduler is set based on the ratio of successive rung
  levels.
* ``type``: The most important values are ``"stopping", "promotion"`` (see
  above).
* ``brackets``: Number of brackets to be used in Hyperband. More details are
  found
  `here <tutorials/multifidelity/mf_asha.html#asynchronous-hyperband>`__.
  The default is 1 (successive halving).

Depending on the searcher, this scheduler supports:

* `Asynchronous successive halving (ASHA) <../multifidelity/mf_asha.html>`__
  [``searcher="random"``]
* `MOBSTER <../multifidelity/mf_async_model.html#asynchronous-mobster>`__
  [``searcher="bayesopt"``]
* `Asynchronous BOHB <../multifidelity/mf_async_model.html#asynchronous-mobster>`__
  [``searcher="kde"``]
* `Hyper-Tune <../multifidelity/mf_async_model.html#hyper-tune>`__
  [``searcher="hypertune"``]
* Cost-aware Bayesian optimization [``searcher="bayesopt_cost"``]
* Bore [``searcher="bore"``]
* DyHPO [``searcher="dyhpo", type="dyhpo"``]

We will only consider the first two searchers in this tutorial.

Asynchronous Hyperband (ASHA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` is configured
with a random searcher, we obtain ASHA, as proposed in
`A System for Massively Parallel Hyperparameter Tuning <https://arxiv.org/abs/1810.05934>`__.
More details are provided `here <tutorials/multifidelity/mf_asha.html>`__.
Nothing much can be configured via ``search_options`` in this case. The
arguments are the same as for random search with ``FIFOScheduler``.

Model-based Asynchronous Hyperband (MOBSTER)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` is configured with
a Bayesian optimization searcher, we obtain MOBSTER, as proposed in
`Model-based Asynchronous Hyperparameter and Neural Architecture Search <https://openreview.net/forum?id=a2rFihIU7i>`__.
By default, MOBSTER uses a multi-task Gaussian process surrogate model for
metrics data observed at all resource levels. More details are provided
`here <tutorials/multifidelity/mf_async_model.html#asynchronous-mobster>`__.

Recommendations
---------------

Finally, we provide some general recommendations on how to use our built-in
schedulers.

* If you can afford it for your problem, random search is a useful baseline
  (:class:`~syne_tune.optimizer.baselines.RandomSearch`). However, if even a
  single full evaluation takes a long time, try ASHA
  (:class:`~syne_tune.optimizer.baselines.ASHA`) instead. The default for ASHA
  is ``type="stopping"``, but you should consider ``type="promotion"`` as well
  (more details on this choice are given
  `here <tutorials/multifidelity/mf_asha.html#asynchronous-successive-halving-promotion-variant>`__.
* Use these baseline runs to get an idea how long your experiment needs to run.
  It is recommended to use a stopping criterion of the form
  ``stop_criterion=StoppingCriterion(max_wallclock_time=X)``, so that the
  experiment is stopped after ``X`` seconds.
* If your tuning problem comes with an obvious resource parameter, make sure to
  implement it such that results are reported during the evaluation, not only
  at the end. When training a neural network model, choose the number of epochs
  as resource. In other situations, choosing a resource parameter may be more
  difficult. Our schedulers require positive integers. Make sure that
  evaluations for the same configuration scale linearly in the resource
  parameter: an evaluation up to ``2 * r`` should be roughly twice as
  expensive as one up to ``r``.
* If your problem has a resource parameter, always make sure to try
  :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`, which in many
  cases runs much faster than
  :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`.
* If you end up tuning the same ML algorithm or neural network model on
  different datasets, make sure to set ``points_to_evaluate`` appropriately. If
  the model comes from frequently used open source code, its built-in defaults
  will be a good choice. Any hyperparameter not covered in
  ``points_to_evaluate`` is set using a midpoint heuristic. While still better
  than choosing the first configuration at random, this may not be very good.
* In general, the defaults should work well if your tuning problem is expensive
  enough (at least a minute per unit of ``r``). In such cases, MOBSTER
  (:class:`~syne_tune.optimizer.baselines.MOBSTER`) can outperform ASHA
  substantially. However, if your problem is cheap, so you can afford a lot of
  evaluations, the searchers based on GP surrogate models may end up expensive.
  In fact, once the number of evaluations surpassed a certain threshold, the
  data is filtered down before fitting the surrogate model (see
  `here <tutorials/multifidelity/mf_async_model.html#asynchronous-mobster>`__).
  You can adjust this threshold or change ``opt_skip_period`` in order to speed
  up MOBSTER.
