Benchmarking with Simulator Backend
===================================

The fastest and cheapest way to compare a number of different HPO methods, or
variants thereof, is benchmarking with the simulator backend. In this case,
all training evaluations are simulated by querying metric and time values from
a tabulated blackbox or a surrogate model. Not only are expensive computations
on GPUs avoided, but the experiment also runs faster than real time. In some
cases, results for experiments with ``max_wallclock_time`` of several hours,
can be obtained in a few seconds.

Defining the Experiment
-----------------------

As usual in Syne Tune, the experiment is defined by a number of scripts. We
will look at an example in
`benchmarking/nursery/benchmark_hypertune/ <../../benchmarking/benchmark_hypertune.html>`__.
Common code used in these benchmarks can be found in :mod:`benchmarking.commons`.

* Local launcher: :mod:`benchmarking.commons.hpo_main_simulator`
* Remote launcher: :mod:`benchmarking.commons.launch_remote_simulator`
* Benchmark definitions: :mod:`benchmarking.commons.benchmark_definitions`

Let us look at the scripts in order, and how you can adapt them to your needs:

* `benchmarking/nursery/benchmark_hypertune/baselines.py <../../benchmarking/benchmark_hypertune.html#id1>`__:
  Defines the HPO methods to take part in the experiment, in the form of a
  dictionary ``methods`` which maps method names to factory functions, which in
  turn map :class:`~benchmarking.commons.baselines.MethodArguments` to scheduler
  objects. The :class:`~benchmarking.commons.baselines.MethodArguments` class
  contains the union of attributes needed to configure schedulers. In particular,
  ``scheduler_kwargs`` contains constructor arguments. For your convenience, the
  mapping from ``MethodsArguments`` to scheduler are defined for most baseline
  methods in :mod:`benchmarking.commons.default_baselines` (as noted just below,
  this mapping involves merging argument dictionaries), but you can override
  arguments as well (for example, ``type`` in the examples
  here). Note that if you like to compare different variants of a method, you
  need to create different entries in ``methods``, for example
  ``Methods.MOBSTER_JOINT`` and ``Methods.MOBSTER_INDEP`` are different variants
  of MOBSTER.
* `benchmarking/nursery/benchmark_hypertune/benchmark_definitions.py <../../benchmarking/benchmark_hypertune.html#id2>`__:
  Defines the benchmarks to be considered in this experiment, in the form of a
  dictionary ``benchmark_definitions`` with values of type
  :class:`~benchmarking.commons.benchmark_definitions.SurrogateBenchmarkDefinition`.
  In general, you will just pick definitions from
  :mod:`benchmarking.commons.benchmark_definitions`. But you can also modify
  parameters, for example ``surrogate`` and ``surrogate_kwargs`` in order to
  select a different surrogate model, or you can change the defaults for
  ``n_workers`` or ``max_wallclock_time``.
* `benchmarking/nursery/benchmark_hypertune/hpo_main.py <../../benchmarking/benchmark_hypertune.html#id3>`__:
  Script for launching experiments locally. All you typically need to do here
  is to import :mod:`benchmarking.commons.hpo_main_simulator` and (optionally)
  to add additional command line arguments you would like to parameterize your
  experiment with. In our example here, we add two options, ``num_brackets``
  which configures Hyperband schedulers, and ``num_samples`` which configures
  the Hyper-Tune methods only. Apart from ``extra_args``, you also need to
  define ``map_extra_args``, which modifies ``method_kwargs`` (the arguments of
  :class:`~benchmarking.commons.baselines.MethodArguments`) based on the extra
  arguments. Details for ``map_extra_kwargs`` are given just below. Finally,
  :func:`~benchmarking.commons.hpo_main_simulator.main` is called with your
  ``methods`` and ``benchmark_definitions`` dictionaries, and (optionally) with
  ``extra_args`` and ``map_extra_args``. We will see shortly how the launcher
  is called, and what happens inside.
* `benchmarking/nursery/benchmark_hypertune/launch_remote.py <../../benchmarking/benchmark_hypertune.html#id4>`__:
  Script for launching experiments remotely, in that each experiment runs as its
  own SageMaker training job, in parallel with other experiments. You need to
  import :mod:`benchmarking.commons.launch_remote_simulator` and pass the same
  ``methods``, ``benchmark_definitions``, ``extra_args`` as
  in :mod:`benchmarking.nursery.benchmark_hypertune.hpo_main`. On top of that,
  you can pass an indicator function ``is_expensive_method`` to tag the HPO
  methods which are themselves expensive to run. As detailed below, our script
  runs different seeds (repetitions) in parallel for expensive methods, but
  sequentially for cheap ones. We will see shortly how the launcher is called,
  and what happens inside.
* `benchmarking/nursery/benchmark_hypertune/requirements.txt <../../benchmarking/benchmark_hypertune.html#id5>`__:
  Dependencies for ``hpo_main.py`` to be run remotely as SageMaker training job,
  in the context of launching experiments remotely. In particular, this needs
  the dependencies of Syne Tune itself. A safe bet here is ``syne-tune[extra]``
  and ``tqdm`` (which is the default if ``requirements.txt`` is missing). However,
  you can decrease startup time by narrowing down the dependencies you really
  need (see
  `FAQ <../../faq.html#what-are-the-different-installations-options-supported>`__).
  In our example here, we need ``gpsearchers`` and ``kde`` for methods. For
  simulated experiments, you always need to have ``blackbox-repository`` here.
  In order to use YAHPO benchmarks, also add ``yahpo``.

Specifying Extra Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~

In many cases, you will want to run different methods using their default
arguments, or only change them as part of the definition in ``baselines.py``.
But sometimes, it can be useful to be able to set options via extra command line
arguments. This can be done via ``extra_args`` and ``map_extra_args``, which are
typically used in order to be able to configure scheduler arguments for certain
methods. But in principle, any argument of
:class:`~benchmarking.commons.baselines.MethodArguments` can be modified. Here,
``extra_args`` is simply extending arguments to the command line parser, where the
``name`` field contains the name of the option without any leading "-".
``map_extra_args`` has the signature

.. code-block:: python

   method_kwargs = map_extra_args(args, method, method_kwargs)

Here, ``method_kwargs`` are arguments of
:class:`~benchmarking.commons.baselines.MethodArguments`, which can be modified
by ``map_extra_kwargs`` (the modified dictionary is returned). ``args`` is the
result of command line parsing, and ``method`` is the name of the method to
be constructed based on these arguments. The latter argument allows
``map_extra_kwargs`` to depend on the method. In our example
`benchmarking/nursery/benchmark_hypertune/hpo_main.py <../../benchmarking/benchmark_hypertune.html#id3>`__,
``num_brackets`` applies to all methods, while ``num_samples`` only applies
to the variants of Hyper-Tune. Both arguments modify the dictionary
``scheduler_kwargs`` in :class:`~benchmarking.commons.baselines.MethodArguments`,
which contains constructor arguments for the scheduler.

Note the use of ``recursive_merge``. This means that the changes done in
``map_extra_args`` are recursively merged into the prior ``method_kwargs``. In
our example, we may already have ``method_kwargs.scheduler_kwargs`` or even
``method_kwargs.scheduler_kwargs.search_options``. While the new settings here
take precedence, prior content of ``method_kwargs`` not affected remains in
place. In the same way, extra arguments passed to baseline wrappers in
:mod:`benchmarking.commons.default_baselines` are recursively merged into the
arguments determined by the default logic.

Writing Extra Results
~~~~~~~~~~~~~~~~~~~~~

By default, Syne Tune writes result files ``metadata.json``, ``results.csv.zip``,
and ``tuner.dill`` for every experiment, see
`here <../../faq.html#what-does-the-output-of-the-tuning-contain>`__. Here,
``results.csv.zip`` contains all data reported by training jobs, along with
time stamps. The contents of this dataframe can be customized, by adding extra
columns to it. This is done by passing ``extra_results_composer`` of type
:class:`~syne_tune.results_callback.ExtraResultsComposer` when creating the
:class:`~syne_tune.results_callback.StoreResultsCallback` callback, which
is passed in ``callbacks`` to :class:`~syne_tune.Tuner`. You can use this
mechanism by passing a :class:`~syne_tune.results_callback.ExtraResultsComposer`
object as ``extra_results`` to ``main``. This object extracts extra information
and returns it as dictionary, which is appended to the results dataframe. A
complete example is
``benchmarking/nursery/benchmark_dyhpo <../../benchmarking/benchmark_dyhpo.html>`__.

Launching Experiments Locally
-----------------------------

Here is an example of how simulated experiments are launched locally:

.. code-block:: bash

   python benchmarking/nursery/benchmark_hypertune/hpo_main.py \
     --experiment_tag tutorial-simulated --benchmark nas201-cifar100 \
     --method ASHA --num_seeds 10

This call runs a number of experiments sequentially on the local machine:

* ``experiment_tag``: Results of experiments are written to
  ``~/syne-tune/{experiment_tag}/*/{experiment_tag}-*/``. This name should
  confirm to S3 conventions (alphanumerical and ``-``; no underscores).
* ``benchmark``: Selects benchmark from keys of ``benchmark_definitions``.
  If this is not given, experiments for all keys in ``benchmark_definitions``
  are run in sequence.
* ``method``: Selects HPO method to run from keys of ``methods``. If this is
  not given, experiments for all keys in ``methods`` are run in sequence.
* ``num_seeds``: Each experiment is run ``num_seeds`` times with different
  seeds (``0, ..., num_seeds - 1``). Due to random factors both in training
  and tuning, a robust comparison of HPO methods requires such repetitions.
  Fortunately, these are cheap to obtain in the simulation context. Another
  parameter is ``start_seed`` (default: 0), giving seeds
  ``start_seed, ..., num_seeds - 1``. For example, ``--start_seed 5  --num_seeds 6``
  runs for a single seed equal to 5. The dependence of random choices on the
  seed is detailed `below <bm_local.html#random-seeds-and-paired-comparisons>`__.
* ``max_wallclock_time``, ``n_workers``: These arguments overwrite the defaults
  specified in the benchmark definitions.
* ``max_size_data_for_model``: Parameter for Bayesian optimization, MOBSTER or
  Hyper-Tune, see
  `here <../multifidelity/mf_async_model.html#controlling-mobster-computations>`__
  and
  `here <../basics/basics_bayesopt.html#speeding-up-decision-making>`__.
* ``scale_max_wallclock_time``: If 1, and if ``n_workers`` is given as
  argument, but not ``max_wallclock_time``, the benchmark default
  ``benchmark.max_wallclock_time`` is multiplied by :math:``B / min(A, B)``,
  where ``A = n_workers``, ``B = benchmark.n_workers``. This means we run for
  longer if ``n_workers < benchmark.n_workers``, but keep
  ``benchmark.max_wallclock_time`` the same otherwise.
* ``use_long_tuner_name_prefix``: If 1, results for an experiment are written
  to a directory whose prefix is
  :code:`f"{experiment_tag}-{benchmark_name}-{seed}"`, followed by a postfix
  containing date-time and a 3-digit hash. If 0, the prefix is
  :code:`experiment_tag` only. The default is 1 (long prefix).
* ``restrict_configurations``: See
  `below <#restricting-scheduler-to-configurations-of-tabulated-blackbox>`__.
* ``fcnet_ordinal``: Applies to FCNet benchmarks only. The hyperparameter
  ``hp_init_lr`` has domain ``choice([0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])``.
  Since the parameter is really ordinal, this is
  `not a good choice <../../search_space.html#recommendations>`__. With this
  option, the domain can be switched to different variants of ``ordinal``.
  The default is ``nn-log``, which is the domain
  ``logordinal([0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])``. In order to keep
  the original categorical domain, use ``--fcnet_ordinal none``.

If you defined additional arguments via ``extra_args``, you can use them
here as well. For example, ``--num_brackets 3`` would run all
multi-fidelity methods with 3 brackets (instead of the default 1).

Launching Experiments Remotely
------------------------------

There are some drawbacks of launching experiments locally. First, they block
the machine you launch from. Second, different experiments are run sequentially,
not in parallel. Remote launching has exactly the same parameters as launching
locally, but experiments are sliced along certain axes and run in parallel,
using a number of SageMaker training jobs. Here is an example:

.. code-block:: bash

   python benchmarking/nursery/benchmark_hypertune/launch_remote.py \
     --experiment_tag tutorial-simulated --benchmark nas201-cifar100 \
     --num_seeds 10

Since ``--method`` is not used, we run experiments for all methods. Also, we
run experiments for 10 seeds. There are 7 methods, so the total number of
experiments is 70 (note that we select a single benchmark here). Running this
command will launch 43 SageMaker training jobs, which do the work in parallel.
Namely, for methods ``ASHA``, ``SYNCHB``, ``BOHB``, all 10 seeds are run
sequentially in a single SageMaker job, since our ``is_expensive_method``
function returns ``False`` for them. Simulating experiments is so fast for
these methods that it is best to run seeds sequentially. However, for
``MOBSTER-JOINT``, ``MOBSTER-INDEP``, ``HYPERTUNE-INDEP``, ``HYPERTUNE-JOINT``,
our ``is_expensive_method`` returns ``True``, and we use one SageMaker
training jobs for each seeds, giving rise to ``4 * 10 = 40`` jobs running in
parallel. For these methods, the simulation time is quite a bit longer, because
decision making takes more time (these methods fit Gaussian process surrogate
models to data and optimize acquisition functions). Results are written to
``~/syne-tune/{experiment_tag}/ASHA/`` for the cheap method ``ASHA``, and to
``/syne-tune/{experiment_tag}/MOBSTER-INDEP-3/`` for the expensive method
``MOBSTER-INDEP`` and seed 3.

The command above selected a single benchmark ``nas201-cifar100``. If
``--benchmark`` is not given, we iterate over all benchmarks in
``benchmark_definitions``. This is done sequentially, which works fine for a
limited number of benchmarks.

However, you may want to run experiments on a large number of benchmarks, and
to this end also parallelize along the benchmark axis. To do so, you can pass
a nested dictionary as ``benchmark_definitions``. For example, we could use the
following:

.. code-block:: python

   from benchmarking.commons.benchmark_definitions import (
       nas201_benchmark_definitions,
       fcnet_benchmark_definitions,
       lcbench_selected_benchmark_definitions,
   )

   benchmark_definitions = {
       "nas201": nas201_benchmark_definitions,
       "fcnet": fcnet_benchmark_definitions,
       "lcbench": lcbench_selected_benchmark_definitions,
   }

In this case, experiments are sliced along the axis
``("nas201", "fcnet", "lcbench")`` to be run in parallel in different SageMaker
training jobs.

Pitfalls of Experiments from Tabulated Blackboxes
-------------------------------------------------

Comparing HPO methods on tabulated benchmarks, using simulation, has obvious
benefits. Costs are very low. Moreover, results are often obtain many times
faster than real time. However, we recommend you do not rely on such kind of
benchmarking only. Here are some pitfalls:

* Tabulated benchmarks are often of limited complexity, because more complex
  benchmarks cannot be sampled exhaustively
* Tabulated benchmarks do not reflect the stochasticity of real benchmarks
  (e.g., random weight initialization, random ordering of mini-batches)
* While tabulated benchmarks like ``nas201`` or ``fcnet`` are evaluated
  exhaustively or on a fine grid, other benchmarks (like ``lcbench``) contain
  observations only at a set of randomly chosen configurations, while their
  configuration space is much larger or even infinite. For such benchmarks,
  you can either restrict the scheduler to suggest configurations only from
  the set supported by the benchmark (see subsection just below), or you can
  use a surrogate model which interpolates observations from those contained
  in the benchmark to all others in the configuration space. Unfortunately, the
  choice of surrogate model can strongly affect the benchmark, for the same
  underlying data. As a general recommendation, you should be careful with
  surrogate benchmarks which offer a large configuration space, but are based
  on only medium amounts of real data.

Restricting Scheduler to Configurations of Tabulated Blackbox
-------------------------------------------------------------

For a tabulated benchmark like ``lcbench``, most entries of the configuration
space are not covered by data. For such, you can either use a surrogate, which
can be configured by attributes ``surrogate``, ``surrogate_kwargs``, and
``add_surrogate_kwargs`` of
:class:`~benchmarking.commons.benchmark_definitions.SurrogateBenchmarkDefinition`.
Or you can restrict the scheduler to only suggest configurations covered by
data. The latter is done by the option ``--restrict_configurations 1``. The
advantage of doing so is that your comparison does not depend on the choice of
surrogate, but only on the benchmark data itself. However, there are also some
drawbacks:

* This option is currently not supported for the following schedulers:

  * Grid Search
  * SyncBOHB
  * BOHB
  * DEHB
  * REA
  * KDE
  * PopulationBasedTraining
  * ZeroShotTransfer
  * ASHACTS
  * MOASHA

* Schedulers like Gaussian process based Bayesian optimization typically use
  local gradient-based optimization of the acquisition function. This is not
  possible with ``--restrict_configurations 1``. Instead, they evaluate the
  acquisition function at a finite number ``num_init_candidates`` of points and
  pick the best one
* In general, you should avoid to use surrogate benchmarks which offer a large
  configuration space, but are based on only medium amounts of real data. When
  using ``--restrict_configurations 1`` with such a benchmark, your methods
  may perform better than they should, just because they nearly sample the
  space exhaustively

Selecting Benchmarks from benchmark_definitions
-----------------------------------------------

Each family of tabulated (or surrogate) blackboxes accessible to the
benchmarking tooling discussed here, are represented by a Python file in
:mod:`benchmarking.commons.benchmark_definitions` (the same directly also
contains definitions for `real benchmarks <bm_local.html>`__). For example:

* NASBench201 (:mod:`benchmarking.commons.benchmark_definitions.nas201`):
  Tabulated, no surrogate needed.
* FCNet (:mod:`benchmarking.commons.benchmark_definitions.fcnet`):
  Tabulated, no surrogate needed.
* LCBench (:mod:`benchmarking.commons.benchmark_definitions.lcbench`):
  Needs surrogate model (scikit-learn regressor) to be selected.
* YAHPO (:mod:`benchmarking.commons.benchmark_definitions.yahpo`):
  Contains a number of blackboxes, some with a large number of instances.
  All these are surrogate benchmarks, with a special surrogate model.

Typically, a blackbox concerns a certain machine learning algorithm with a fixed
configuration space. Many of them have been evaluated over a number of
different datasets. Note that in YAHPO, a *blackbox* is called *scenario*, and
a *dataset* is called *instance*, so that a scenario can have a certain number
of instances. In our terminology, a tabulated *benchmark* is obtained by
selecting a blackbox together with a dataset.

The files in :mod:`benchmarking.commons.benchmark_definitions` typically
contain:

* Functions named ``*_benchmark``, which map arguments (such as ``dataset_name``)
  to the benchmark definition
  :class:`~benchmarking.commons.benchmark_definitions.SurrogateBenchmarkDefinition`
  and ``*`` being the name of the blackbox (or scenario).
* Dictionaries named ``*_benchmark_definitions`` with
  :class:`~benchmarking.commons.benchmark_definitions.SurrogateBenchmarkDefinition`
  values. If a blackbox has a lot of datasets, we also define a dictionary
  ``*_selected_benchmark_definitions``, which selects benchmarks which are
  interesting (e.g., not all baselines achieving the same performance rapidly).
  In general, we recommend starting with these selected benchmarks.

The YAHPO Family
~~~~~~~~~~~~~~~~

A rich source of blackbox surrogates in Syne Tune comes from
`YAHPO <https://github.com/slds-lmu/yahpo_gym>`__, which is also detailed in
this `paper <https://arxiv.org/abs/2109.03670>`__. YAHPO contains a number of
blackboxes (called scenarios), some of which over a lot of datasets (called
instances). All our definitions are in
:mod:`benchmarking.commons.benchmark_definitions.yahpo`. Further details can
also be found in the import code
:mod:`syne_tune.blackbox_repository.conversion_scripts.scripts.yahpo_import`.
Here is an overview:

* ``yahpo_nb301``: NASBench301. Single scenario and instance.
* ``yahpo_lcbench``: LCBench. Same underlying data than our own LCBench, but
  different surrogate model.
* ``yahpo_iaml``: Family of blackboxes, parameterized by ML method
  (``yahpo_iaml_methods``) and target metric (``yahpo_iaml_metrics``). Each of
  th`ese have 4 datasets (OpenML datasets).
* ``yahpo_rbv2``: Family of blackboxes, parameterized by ML method
  (``yahpo_rbv2_methods``) and target metric (``yahpo_rbv2_metrics``). Each of
  these come with a large number of datasets (OpenML datasets). Note that
  compared to YAHPO Gym, we filtered out scenarios which are invalid (e.g., F1
  score 0, AUC/F1 equal to 1). We also determined useful ``max_wallclock_time``
  values (``yahpo_rbv2_max_wallclock_time``), and selected benchmarks which
  show interesting behaviour (``yahpo_rbv2_selected_instances``).

.. note::
   At present (YAHPO Gym v1.0), the ``yahpo_lcbench`` surrogate has been
   trained on invalid LCBench original data (namely, values for first and last
   fidelity value have to be removed). As long as this is not fixed, we
   recommend using our built-in ``lcbench`` blackbox instead.

.. note::
   In YAHPO Gym, ``yahpo_iaml`` and ``yahpo_rbv2`` have a fidelity attribute
   ``trainsize`` with values between ``1/20`` and ``1``, which is the fraction
   of full dataset the method has been trained. Our import script multiplies
   ``trainsize`` values with 20 and designates type ``randint(1, 20)``, since
   common Syne Tune multi-fidelity schedulers require ``resource_attr`` values
   to be positive integers. ``yahpo_rbv2`` has a second fidelity attribute
   ``repl``, whose value is constant 10, this is removed by our import script.
