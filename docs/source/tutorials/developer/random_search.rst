Random Search
=============

Random search is arguably the simplest HPO baseline. In a nutshell, ``_suggest``
samples a new configuration at random from the configuration space, much like
our ``SimpleScheduler`` above, and ``on_trial_result`` does nothing except
returning ``SchedulerDecision.CONTINUE``. A slightly more advanced version
would make sure that the same configuration is not suggested twice.

In this section, we walk through the Syne Tune implementation of random search,
thereby discussing some additional concepts. This will also be a first example
of the modular concept just described: random search is implemented as generic
:class:`~syne_tune.optimizer.schedulers.FIFOScheduler` configured by a
:class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`.
A self-contained implementation of random search would be shorter. On the other
hand, as seen in
:mod:`syne_tune.optimizer.baselines`, ``FIFOScheduler`` also powers GP-based
Bayesian optimization, grid search, BORE, regularized evolution and constrained
BO simply by specifying different searchers. A number of concepts, to be
discussed here, have to be implemented once only and can be maintained much more
easily.

FIFOScheduler and RandomSearcher
--------------------------------

We will have a close look at
:class:`~syne_tune.optimizer.schedulers.FIFOScheduler` and
:class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`. Let us first
consider the arguments of ``FIFOScheduler``:

* ``searcher``, ``search_options``: These are used to configure the scheduler
  with a searcher. For ease of use, ``searcher`` can be a name, and additional
  arguments can be passed via ``search_options``. In this case, the searcher is
  created by a factory, as detailed `below <new_searcher.html>`__. Alternatively,
  ``searcher`` can also be a
  :class:`~syne_tune.optimizer.schedulers.searchers.BaseSearcher` object.
* ``metric``, ``mode``: As discussed `above <first_example.html#first-example>`__
  in ``SimpleScheduler``.
* ``random_seed``: Several pseudo-random number generators may be used in
  scheduler and searcher. Seeds for these are drawn from a random seed generator
  maintained in :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`, whose
  seed can be passed here. As a general rule, all schedulers and searchers
  implemented in Syne Tune carefully manage such generators (and contributed
  schedulers are strongly encourage to adopt this pattern).
* ``points_to_evaluate``: A list of configurations (possibly partially specified)
  to be suggested first. This allows the user to initialize the search by
  default configurations, thereby injecting knowledge about the task. We
  strongly recommend every scheduler to support this mechanism. More details
  are given below.
* ``max_resource_attr``, ``max_t``: These arguments are relevant for
  multi-fidelity schedulers. Only one of them needs to be given. We recommend
  to use ``max_resource_attr``. More details are given
  `below <extend_async_hb.html#hyperbandscheduler>`__.

The most important use case is to configure ``FIFOScheduler`` with a new
searcher, and we will concentrate on this one. First, the base class of all
searchers is :class:`~syne_tune.optimizer.schedulers.searchers.BaseSearcher`:

* ``points_to_evaluate``: A list of configurations to be suggested first. This
  is initialized and (possibly) imputed in the base class, but needs to be used
  in child classes. Configurations in ``points_to_evaluate`` can be partially
  specified. Any hyperparameter missing in a configuration is imputed using a
  “midpoint” rule. For a numerical parameter, this is the middle of the range
  (in linear or log scale). For a categorical parameter, the first value is
  chosen. If ``points_evaluate`` is not given, the default is ``[dict()]``: a
  single initial configuration is determined fully by the midpoint rule. In
  order not to use initial configurations at all, the user has to pass
  ``points_to_evaluate=[]``. The imputation of configurations is done in the
  base class.
* ``configure_scheduler``: Callback function, allows the searcher to configure
  itself depending on the scheduler. It also allows the searcher to reject
  schedulers it is not compatible with. This method is called automatically at
  the beginning of an experiment.
* ``get_config``: This method is called by the scheduler in ``_suggest``, it
  delegates the suggestion of a configuration for a new trial to the searcher.
* ``on_trial_result``: This is called by the scheduler in its own
  ``on_trial_result``, also passing the configuration of the current trial. If
  the searcher maintains a surrogate model (for example, based on a Gaussian
  process), it should update its model with ``result`` data iff ``update=True``.
  This is discussed in more detail `below <extend_async_hb.html>`__. Note that
  ``on_trial_result`` does not return anything: decisions on how to proceed
  with the trial are not done in the searcher.
* ``register_pending``: Registers one (or more) pending evaluations, which are
  signals to the searcher that a trial has been started and will return an
  observation in the future. This is important in order to avoid redundant
  suggestions in model-based HPO.
* ``evaluation_failed``: Called by the scheduler if a trial failed. Default
  searcher reactions are to remove pending evaluations and not to suggest the
  corresponding configuration again. More advanced constrained searchers may
  also try to avoid nearby configurations in the future.
* ``cleanup_pending``: Removes all pending evaluations for a trial. This is
  called by the scheduler when a trial terminates.
* ``get_state``, ``clone_from_state``: Used in order to serialize and
  de-serialize the searcher
* ``debug_log``: There is some built-in support for a detailed log, embedded in
  ``FIFOScheduler`` and the Syne Tune searchers.

Below ``BaseSearcher``, there is
:class:`~syne_tune.optimizer.schedulers.searchers.StochasticSearcher`, which
should be used by all searchers which make random decisions. It maintains a PRN
generator and provides methods to serialize and de-serialize its state.

:class:`~syne_tune.optimizer.schedulers.searchers.StochasticAndFilterDuplicatesSearcher`
extends ``StochasticSearcher``. It supports a number of features which are
desirable for most searchers:

* Seed management for random decisions.
* Avoid suggesting the same configuration more than once. While we in general
  recommend to use the default ``allow_duplicates == False``, allowing for
  duplicates can be useful when dealing with configuration spaces of small
  finite size.
* Restrict configurations which can be suggested to a finite set. This can be
  very useful when
  `using tabulated blackboxes <../benchmarking/bm_simulator.html#restricting-scheduler-to-configurations-of-tabulated-blackbox>`__.
  It does not make sense for every scheduler though, as some rely on a
  continuous search over the configuration space. You can inherit from
  :class:`~syne_tune.optimizer.schedulers.searchers.StochasticAndFilterDuplicatesSearcher`
  and still not support this feature, by insisting on
  ``restrict_configurations == None``.

All built-in Syne Tune searchers either inherit from this class, or avoid
duplicate suggestions in a different way. Finally, let us walk through
:class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`:

* There are a few features beyond ``SimpleScheduler`` above. The searcher does
  not suggest the same configuration twice (if ``allow_duplicates == False``),
  and also warns if a finite configuration space has been exhausted. It also uses
  :class:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges`
  for random sampling and comparing configurations (to spot duplicates). This
  is a useful helper class, also for encoding configurations as vectors. The
  logic of detecting duplicates is implemented in the base class
  :class:`~syne_tune.optimizer.schedulers.searchers.StochasticAndFilterDuplicatesSearcher`.
  Finally, ``debug_log`` is used for diagnostic logs.
* ``get_config`` first asks for another entry from ``points_to_evaluate`` by
  way of ``_next_initial_config``. It then samples a new configuration at
  random. This is done without replacement if ``allow_duplicates == False``,
  and with replacement otherwise. If successful, it also feeds ``debug_log``.
* ``_update``: This is not needed for random search, but is used here in order
  to feed ``debug_log``.
