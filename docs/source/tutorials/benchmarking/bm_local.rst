Benchmarking with Local Backend
================================

A *real benchmark* (as opposed to a benchmark based on tabulated data or a
surrogate model) is based on a training script, which is executed for each
evaluation. The local backend is the default choice in Syne Tune for running
on real benchmarks.

.. note::
   While Syne Tune contains benchmark definitions for all surrogate benchmarks
   in :mod:`syne_tune.experiments.benchmark_definitions`, examples for real
   benchmarks are only available when Syne Tune is installed from source.
   They are located in :mod:`benchmarking`.

Defining the Experiment
-----------------------

As usual in Syne Tune, the experiment is defined by a number of scripts.
We will look at an example in
`benchmarking/examples/launch_local/ <../../benchmarking/launch_local.html>`__.
Common code used in these benchmarks can be found in
:mod:`syne_tune.experiments`:

* Local launcher: :mod:`syne_tune.experiments.launchers.hpo_main_local`
* Remote launcher: :mod:`syne_tune.experiments.launchers.launch_remote_local`
* Definitions for real benchmarks: :mod:`benchmarking.benchmark_definitions`

Let us look at the scripts in order, and how you can adapt them to your needs:

* `benchmarking/examples/launch_local/baselines.py <../../benchmarking/launch_local.html#id1>`__:
  This is the same as in the
  `simulator case <bm_simulator.html#defining-the-experiment>`__.
* `benchmarking/examples/launch_local/hpo_main.py <../../benchmarking/launch_local.html#id2>`__:
  This is the same as in the
  `simulator case <bm_simulator.html#defining-the-experiment>`__, but based on
  :mod:`syne_tune.experiments.launchers.hpo_main_local`. We will see shortly how the
  launcher is called, and what happens inside.
* `benchmarking/examples/launch_local/launch_remote.py <../../benchmarking/launch_local.html#id3>`__:
  Much the same as in the
  `simulator case <bm_simulator.html#defining-the-experiment>`__, but based on
  :mod:`syne_tune.experiments.launchers.launch_remote_local`. We will see shortly how the
  launcher is called, and what happens inside. Note that
  ``source_dependencies=benchmarking.__path__``, which allows the launcher
  script to access the training code and benchmark definitions.
* `benchmarking/examples/launch_local/requirements-synetune.txt <../../benchmarking/launch_local.html#id4>`__:
  This file is for defining the requirements of the SageMaker training job in
  remote launching, it mainly has to contain the Syne Tune dependencies. Your
  training script may have additional dependencies, and they are combined with
  the ones here automatically, as detailed below.

Extra arguments can be specified by ``extra_args``, ``map_method_args``, and
extra results can be written using ``extra_results``, as is explained
`here <bm_simulator.html#specifying-extra-arguments>`__.

Launching Experiments Locally
-----------------------------

Here is an example of how experiments with the local backend are launched
locally:

.. code-block:: bash

   python benchmarking/examples/launch_local/hpo_main.py \
     --experiment_tag tutorial-local --benchmark resnet_cifar10 \
     --method ASHA --num_seeds 1 --n_workers 1

This call runs a single experiment on the local machine (which needs to have a
GPU with PyTorch being installed):

* ``experiment_tag``: Results of experiments are written to
  ``~/syne-tune/{experiment_tag}/*/{experiment_tag}-*/``. This name should
  confirm to S3 conventions (alphanumerical and ``-``; no underscores).
* ``benchmark``: Selects benchmark from keys of
  :func:`~benchmarking.benchmark_definitions.real_benchmark_definitions`.
  The default is ``resnet_cifar10``.
* ``method``: Selects HPO method to run from keys of ``methods``. If this is
  not given, experiments for all keys in ``methods`` are run in sequence.
* ``num_seeds``: Each experiment is run ``num_seeds`` times with different
  seeds (``0, ..., num_seeds - 1``). Due to random factors both in training and
  tuning, a robust comparison of HPO methods requires such repetitions. Another
  parameter is ``start_seed`` (default: 0), giving seeds
  ``start_seed, ..., num_seeds - 1``. For example, ``--start_seed 5 --num_seeds 6``
  runs for a single seed equal to 5.
* ``n_workers``, ``max_wallclock_time``: You can overwrite the default values
  for the selected benchmark by these command line arguments.
* ``max_size_data_for_model``: Parameter for Bayesian optimization, MOBSTER or
  Hyper-Tune, see
  `here <../multifidelity/mf_async_model.html#controlling-mobster-computations>`__
  and
  `here <../basics/basics_bayesopt.html#speeding-up-decision-making>`__.
* ``num_gpus_per_trial``: If you run on an instance with more than one GPU,
  you can prescribe how many GPUs should be allocated to each trial. The default
  is 1. Note that if the product of ``n_workers`` and ``num_gpus_per_trial`` is
  larger than the number of GPUs on the instance, trials will be delayed.
* ``delete_checkpoints``: If 1, checkpoints of trials are removed whenever they
  are not needed anymore. The default is 0, in that all checkpoints are
  retained.
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

If you defined additional arguments via ``extra_args``, you can use them here
as well.

.. note::
   When launching an experiment locally, you need to be on an instance which
   supports the required computations (e.g., has 1 or more GPUs), and you need
   to have installed all required dependencies, including those of the
   SageMaker framework. In the example above, ``resnet_cifar10`` uses the
   ``PyTorch`` framework, and ``n_workers=4`` by default, which we overwrite by
   ``n_workers=1``: you need to launch on a machine with 1 GPU, and with
   PyTorch being installed and properly setup to run GPU computations. If you
   cannot be bothered with all of this, please consider
   `remote launching <bm_local.html#launching-experiments-remotely>`__ as an
   alternative. On the other hand, you can launch experiments locally without
   using SageMaker (or AWS) at all.

Benchmark Definitions
---------------------

In the example above, we select a benchmark via ``--benchmark resnet_cifar10``.
All currently included real benchmarks are collected in
:func:`~benchmarking.benchmark_definitions.real_benchmark_definitions`,
a function which returns the dictionary of real benchmarks, configured by some
extra arguments. If you are happy with selecting one of these existing benchmarks,
you may safely skip this subsection.

For ``resnet_cifar10``, this selects
:func:`~benchmarking.benchmark_definitions.resnet_cifar10.resnet_cifar10_benchmark`,
which returns meta-data for the benchmark as a
:class:`~syne_tune.experiments.benchmark_definitions.RealBenchmarkDefinition`
object. Here, the argument ``sagemaker_backend`` is ``False`` in our case,
since we use the local backend, and additional ``**kwargs`` override arguments
of ``RealBenchmarkDefinition``. Important arguments are:

* ``script``: Absolute filename of the training script. If your script requires
  additional dependencies on top of the SageMaker framework, you need to
  specify them in ``requirements.txt`` in the same directory.
* ``config_space``: Configuration space, this must include ``max_resource_attr``
* ``metric``, ``mode``, ``max_resource_attr``, ``resource_attr``: Names related
  to the benchmark, either of methods reported (output) or of ``config_space``
  entries (input).
* ``max_wallclock_time``, ``n_workers``, ``max_num_evaluations``: Defaults for
  tuner or stopping criterion, suggested for this benchmark.
* ``instance_type``: Suggested AWS instance type for this benchmark.
* ``framework``, ``estimator_kwargs``: SageMaker framework and additional
  arguments to SageMaker estimator.

Note that parameters like ``n_workers``, ``max_wallclock_time``, or
``instance_type`` are given default values here, which can be overwritten
by command line arguments. This is why the function signature ends with
``**kwargs``, and we execute ``_kwargs.update(kwargs)`` just before creating
the ``RealBenchmarkDefinition`` object.

Launching Experiments Remotely
------------------------------

Remote launching is particularly convenient for experiments with the local
backend, even if you just want to run a single experiment. For local
launching, you need to be on an EC2 instance of the desired instance type, and
Syne Tune has to be installed there along with all dependencies of your
benchmrk. None of this needs to be done for remote launching. Here is an
example:

.. code-block:: bash

   python benchmarking/examples/launch_local/launch_remote.py \
     --experiment_tag tutorial-local --benchmark resnet_cifar10 \
     --num_seeds 5

Since ``--method`` is not used, we run experiments for all methods (``RS``,
``BO``, ``ASHA``, ``MOBSTER``), and for 5 seeds. These are 20 experiments,
which are mapped to 20 SageMaker training jobs. These will run on instances of
type ``ml.g4dn.12xlarge``, which is the default for ``resnet_cifar10`` and the
local backend. Instances of this type have 4 GPUs, so we can use ``n_workers``
up to 4 (the default being 4). Results are written to S3, using paths such as
``syne-tune/{experiment_tag}/ASHA-3/`` for method ``ASHA`` and seed 3.

Finally, some readers may be puzzled why Syne Tune dependencies are defined in
``benchmarking/examples/launch_local/requirements-synetune.txt``, and not in
``requirements.txt`` instead. The reason is that dependencies of the SageMaker
estimator for running the experiment locally is really the union of two such
files. First, ``requirements-synetune.txt`` for the Syne Tune dependencies,
and second, ``requirements.txt`` next to the training script. The remote
launching script is creating a ``requirements.txt`` file with this union in
``benchmarking/examples/launch_local/``, which should not become part of the
repository.

Visualizing Tuning Metrics in the SageMaker Training Job Console
----------------------------------------------------------------

When experiments are launched remotely with the local or SageMaker backend, a
number of metrics are published to the SageMaker training job console (this
feature can be switched off with ``--remote_tuning_metrics 0``):

* :const:`~syne_tune.remote.remote_metrics_callback.BEST_METRIC_VALUE`: Best
  metric value attained so far
* :const:`~syne_tune.remote.remote_metrics_callback.BEST_TRIAL_ID`: ID of trial
  for best metric value so far
* :const:`~syne_tune.remote.remote_metrics_callback.BEST_RESOURCE_VALUE`:
  Resource value for best metric value so far
* :const:`~syne_tune.remote.remote_metrics_callback.BEST_HP_PREFIX`, followed
  by hyperparameter name: Hyperparameter value for best metric value so far

You can inspect these metrics in real time in AWS CloudWatch. To do so:

* Locate the training job running your experiment in the AWS SageMaker console.
  Click on ``Training``, then ``Training jobs``, then on the job in the list.
  For the command above, the jobs are named like
  ``tutorial-local-RS-0-XyK8`` (experiment tag, then method, then seed, then
  4-character hash).
* Under ``Metrics``, you will see a number of entries, starting with
  ``best_metric_value`` and ``best_trial_id``.
* Further below, under ``Monitor``, click on ``View algorithm metrics``. This
  opens a CloudWatch dashboard
* At this point, you need to change a few defaults, in that CloudWatch only
  samples metrics (by grepping the logs) every 5 minutes and then displays
  average values over the 5-minute window. Click on ``Browse`` and select the
  metrics you want to display. For now, select ``best_metric_value``,
  ``best_trial_id``, ``best_resource_value``.
* Click on ``Graphed metrics``, and for every metric, select
  ``Period -> 30 seconds``. Also, select ``Statistics -> Maximum`` for metrics
  ``best_trial_id``, ``best_resource_value``. For ``best_metric_value``, select
  ``Statistics -> Minimum`` if your objective metric is minimized (``mode="min"``),
  and ``Statistics -> Maximum`` otherwise. In our ``resnet_cifar10`` example,
  the objective is accuracy, to be maximized, so we select the latter.
* Finally, select ```10s`` for auto-refresh (the circle with arrow in the
  upper right corner), and change the temporal resolution by displaying ``1h``
  (top row).

This visualization shows you the best metric value attained so far, and which
trial attained it for which resource value (e.g., number of epochs). It can be
improved. For example, we could plot the curves in different axes. Also, we can
visualize the best hyperparameter configuration found so far. In the
``resnet_cifar10`` example, this is given by the metrics ``best_hp_lr``,
``best_hp_batch_size``, ``best_hp_weight_decay``, ``best_hp_momentum``.

Random Seeds and Paired Comparisons
-----------------------------------

Random effects are the most important reason for variations in experimental
outcomes, due to which a meaningful comparison of HPO methods needs to run
a number of repetitions (also called *seeds* above). There are two types of
random effects:

* Randomness in the evaluation of the objective :math:`f(x)` to optimize:
  repeated evaluations of :math:`f` for the same configuration :math:`x`
  result in different metric values.
  In neural network training, these variations originate from random weight
  initialization and the ordering of mini-batches.
* Randomness in the HPO algorithm itself. This is evident for random search
  and ASHA, but just as well concerns Bayesian optimization, since the
  initial configurations are drawn at random, and the optimization of the
  acquisition function involves random choices as well.

Syne Tune allows the second source of randomness to be controlled by passing
a random seed to the scheduler at initialization. If random search is run
several times with the same random seed for the same configuration space,
exactly the same sequence of configurations is suggested. The same holds for ASHA.
When running random search and Bayesian optimization with the same random seed,
the initial configurations (which in BO are either taken from
``points_to_evaluate`` or drawn at random) are identical.

The scheduler random seed used in a benchmark experiment is a combination of
a *master random seed* and the seed number introduced above (the latter has
values :math:`0, 1, 2, \dots`). The master random seed is passed to
``launch_remote.py`` or ``hpo_main.py`` as ``--random_seed``. If no master
random seed is passed, it is drawn at random and output. The master random
seed is also written into ``metadata.json`` as part of experimental results.
Importantly, the scheduler random seed is the same across different methods
for the same seed. This implements a practice called *paired comparison*,
whereby for each seed, different methods are fed with the same random number
sequence. This practice reduces variance between method outcomes, while
still taking account of randomness by running the experiment several times
(for different seeds :math:`0, 1, 2, \dots`).

.. note::
   When comparing several methods on the same benchmark, it is recommended
   to (a) repeat the experiment several times (via ``--num_seeds``), and
   to (b) use the same master random seed. If all comparisons are done
   with a single call of ``launch_remote.py`` or ``hpo_main.py``, this is
   automatically the case, as the master random seed is drawn at random.
   However, if the comparison extends over several calls, make sure to
   note down the master random seed from the first call and pass this
   value via ``--random_seed`` to subsequent calls. The master random seed
   is also stored as ``random_seed`` in the metadata ``metadata.json`` as
   part of experimental results.
