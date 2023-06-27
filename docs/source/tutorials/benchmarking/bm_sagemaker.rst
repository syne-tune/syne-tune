Benchmarking with SageMaker Backend
====================================

The SageMaker backend allows you to run distributed tuning across several
instances, where the number of parallel evaluations is not limited by the
configuration of an instance, but only by your compute budget.

Defining the Experiment
-----------------------

The scripts required to define an experiment are pretty much the same as in the
local backend case. We will look at an example in
`benchmarking/examples/launch_sagemaker/ <../../benchmarking/launch_sagemaker.html>`__.
Common code used in these benchmarks can be found in
:mod:`syne_tune.experiments`:

* Local launcher: :mod:`syne_tune.experiments.launchers.hpo_main_sagemaker`
* Remote launcher: :mod:`syne_tune.experiments.launchers.launch_remote_sagemaker`
* Definitions for real benchmarks: :mod:`benchmarking.benchmark_definitions`

The scripts
`benchmarking/examples/launch_sagemaker/baselines.py <../../benchmarking/launch_sagemaker.html#id1>`__,
`benchmarking/examples/launch_sagemaker/hpo_main.py <../../benchmarking/launch_sagemaker.html#id2>`__, and
`benchmarking/examples/launch_sagemaker/launch_remote.py <../../benchmarking/launch_sagemaker.html#id3>`__
are identical in structure to what happens in the
`local backend case <bm_local.html#defining-the-experiment>`__, with the only
difference that :mod:`syne_tune.experiments.launchers.hpo_main_sagemaker` or
:mod:`syne_tune.experiments.launchers.launch_remote_sagemaker` are imported from. Moreover,
Syne Tune dependencies need to be specified in
`benchmarking/examples/launch_sagemaker/requirements.txt <../../benchmarking/launch_sagemaker.html#id4>`__.

In terms of benchmarks, the same definitions can be used for the SageMaker
backend, in particular you can select from
:func:`~benchmarking.benchmark_definitions.real_benchmark_definitions`.
However, the functions there are called with ``sagemaker_backend=True``, which
can lead to different values in
:class:`~syne_tune.experiments.benchmark_definitions.RealBenchmarkDefinition`.
For example,
:func:`~benchmarking.benchmark_definitions.resnet_cifar10.resnet_cifar10_benchmark`
returns ``instance_type=ml.g4dn.xlarge`` for the SageMaker backend (1 GPU per
instance), but ``instance_type=ml.g4dn.12xlarge`` for the local backend (4 GPUs
per instance). This is because for the local backend to support ``n_workers=4``,
the instance needs to have at least 4 GPUs, but for the SageMaker backend, each
worker uses its own instance, so a cheaper instance type can be used.

Extra arguments can be specified by ``extra_args``, ``map_method_args``, and
extra results can be written using ``extra_results``, as is explained
`here <bm_simulator.html#specifying-extra-arguments>`__.

Launching Experiments Locally
-----------------------------

Here is an example of how experiments with the SageMaker backend are launched
locally:

.. code-block:: bash

   python benchmarking/examples/launch_sagemaker/hpo_main.py \
     --experiment_tag tutorial-sagemaker --benchmark resnet_cifar10 \
     --method ASHA --num_seeds 1

This call launches a single experiment on the local machine (however, each
trial launches the training script as a SageMaker training job, using the
instance type suggested for the benchmark). The command line arguments are the
same as in the
`local backend case <bm_local.html#launching-experiments-locally>`__. Additional
arguments are:

* ``n_workers``, ``max_wallclock_time``: Overwrite the default values for the
  selected benchmark.
* ``max_failures``: Number of trials which can fail without terminating the
  entire experiment.
* ``warm_pool``: This flag is discussed
  `below <bm_sagemaker.html#using-sagemaker-managed-warm-pools>`__.
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

If you defined additional arguments via ``extra_args``, you can use them here
as well.

Launching Experiments Remotely
------------------------------

Sagemaker backend experiments can also be launched remotely, in which case
each experiment is run in a SageMaker training job, using a cheap instance
type, within which trials are executed as SageMaker training jobs as well. The
usage is the same as in the
`local backend case <bm_local.html#launching-experiments-remotely>`__.

When experiments are launched remotely with the SageMaker backend, a number of
metrics are published to the SageMaker training job console (this feature can
be switched off with ``--remote_tuning_metrics 0``). This is detailed
`here <bm_local.html#visualizing-tuning-metrics-in-the-sagemaker-training-job-console>`_.

Using SageMaker Managed Warm Pools
----------------------------------

The SageMaker backend supports
`SageMaker managed warm pools <https://docs.aws.amazon.com/sagemaker/latest/dg/train-warm-pools.html>`__,
a recently launched feature of SageMaker. In a nutshell, this feature allows
customers to circumvent start-up delays for SageMaker training jobs which share
a similar configuration (e.g., framework) with earlier jobs which have already
terminated. For Syne Tune with the SageMaker backend, this translates to
experiments running faster or, for a fixed ``max_wallclock_time``, running more
trials. Warm pools are used if the command line argument ``--warm_pool 1`` is
used with ``hpo_main.py``. For the example above:

.. code-block:: bash

   python benchmarking/examples/launch_sagemaker/hpo_main.py \
     --experiment_tag tutorial-sagemaker --benchmark resnet_cifar10 \
     --method ASHA --num_seeds 1 --warm_pool 1

The warm pool feature is most useful with multi-fidelity HPO methods (such as
``ASHA`` and ``MOBSTER`` in our example). Some points you should be aware of:

* When using SageMaker managed warm pools with the SageMaker backend, it is
  important to use ``start_jobs_without_delay=False`` when creating the
  :class:`~syne_tune.Tuner`.
* Warm pools are a billable resource, and you may incur extra costs arising
  from the fact that up to ``n_workers`` instances are kept running for about
  10 minutes at the end of your experiment. You have to request warm pool quota
  increases for instance types you would like to use. For our example, you need
  to have quotas for (at least) four ``ml.g4dn.xlarge`` instances, **both** for
  training and warm pool usage.
* As a sanity check, you can watch the training jobs in the console. You
  should see ``InUse`` and ``Reused`` in the *Warm pool status* column.
  Running the example above, the first 4 jobs should complete in about 7 to 8
  minutes, while all subsequent jobs should take only 2 to 3 minutes.
