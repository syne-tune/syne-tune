Benchmarking with Local Backend
================================

A *real benchmark* (as opposed to a benchmark based on tabulated data or a
surrogate model) is based on a training script, which is executed for each
evaluation. The local backend is the default choice in Syne Tune for running
on real benchmarks.

Defining the Experiment
-----------------------

As usual in Syne Tune, the experiment is defined by a number of scripts.
We will look at an example in
`benchmarking/nursery/launch_local/ <../../benchmarking/launch_local.html>`_.
Common code used in these benchmarks can be found in
:mod:`benchmarking.commons`:

* Local launcher: :mod:`benchmarking.commons.hpo_main_local`
* Remote launcher: :mod:`benchmarking.commons.launch_remote_local`
* Benchmark definitions: :mod:`benchmarking.commons.benchmark_definitions`

Let us look at the scripts in order, and how you can adapt them to your needs:

* `benchmarking/nursery/launch_local/baselines.py <../../benchmarking/launch_local.html#id1>`_:
  This is the same as in the
  `simulator case <bm_simulator.html#defining-the-experiment>`_.
* `benchmarking/nursery/launch_local/hpo_main.py <../../benchmarking/launch_local.html#id2>`_:
  This is the same as in the
  `simulator case <bm_simulator.html#defining-the-experiment>`_, but based on
  :mod:`benchmarking.commons.hpo_main_local`. We will see shortly how the
  launcher is called, and what happens inside.
* `benchmarking/nursery/launch_local/launch_remote.py <../../benchmarking/launch_local.html#id3>`_:
  Much the same as in the
  `simulator case <bm_simulator.html#defining-the-experiment>`_, but based on
  :mod:`benchmarking.commons.launch_remote_local`. We will see shortly how the
  launcher is called, and what happens inside.
* `benchmarking/nursery/launch_local/requirements-synetune.txt <../../benchmarking/launch_local.html#id4>`_:
  This file is for defining the requirements of the SageMaker training job in
  remote launching, it mainly has to contain the Syne Tune dependencies. Your
  training script may have additional dependencies, and they are combined with
  the ones here automatically, as detailed below.

Launching Experiments Locally
-----------------------------

Here is an example of how experiments with the local backend are launched
locally:

.. code-block:: bash

   python benchmarking/nursery/launch_local/hpo_main.py \
     --experiment_tag tutorial_local --benchmark resnet_cifar10 \
     --method ASHA --num_seeds 1 --n_workers 1

This call runs a single experiment on the local machine (which needs to have a
GPU with PyTorch being installed):

* ``experiment_tag``: Results of experiments are written to
  ``~/syne-tune/{experiment_tag}/*/{experiment_tag}-*/``. This name should
  confirm to S3 conventions (alphanumerical and ``-``; no underscores).
* ``benchmark``: Selects benchmark from keys of
  :func:`~benchmarking.commons.benchmark_definitions.real_benchmark_definitions`.
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
* ``max_size_data_for_model``: Parameter for MOBSTER or Hyper-Tune, see
  `here <../multifidelity/mf_async_model.html#controlling-mobster-computations>`_.

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
   `remote launching <bm_local.html#launching-experiments-remotely>`_ as an
   alternative. On the other hand, you can launch experiments locally without
   using SageMaker (or AWS) at all.

Benchmark Definitions
---------------------

In the example above, we select a benchmark via ``--benchmark resnet_cifar10``.
All currently supported real benchmarks are collected in
:func:`~benchmarking.commons.benchmark_definitions.real_benchmark_definitions`,
a function which returns the dictionary of real benchmarks, configured by some
extra arguments. If you are happy with selecting one of these existing benchmarks,
you may safely skip this subsection.

For ``resnet_cifar10``, this selects
:func:`~benchmarking.commons.benchmark_definitions.resnet_cifar10.resnet_cifar10_benchmark`,
which returns meta-data for the benchmark as a
:class:`~benchmarking.commons.benchmark_definitions.RealBenchmarkDefinition`
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

Note that parameters like ``n_workers`` and ``max_wallclock_time`` are defaults,
which can be overwritten by command line arguments.

Launching Experiments Remotely
------------------------------

Remote launching is particularly convenient for experiments with the local
backend, even if you just want to run a single experiment. For local
launching, you need to be on an EC2 instance of the desired instance type, and
Syne Tune has to be installed there. None of this needs to be done for remote
launching. Here is an example:

.. code-block:: bash

   python benchmarking/nursery/launch_local/launch_remote.py \
     --experiment_tag tutorial_local --benchmark resnet_cifar10 \
     --num_seeds 5

Since ``--method`` is not used, we run experiments for all methods (``RS``,
``BO``, ``ASHA``, ``MOBSTER``), and for 5 seeds. These are 20 experiments,
which are mapped to 20 SageMaker training jobs. These will run on instances of
type ``ml.g4dn.12xlarge``, which is the default for ``resnet_cifar10`` and the
local backend. Instances of this type have 4 GPUs, so we can use ``n_workers``
up to 4 (the default being 4). Results are written to S3, using paths such as
``syne-tune/{experiment_tag}/ASHA-3/`` for method ``ASHA`` and seed 3.

Finally, some readers may be puzzled why Syne Tune dependencies are defined in
``benchmarking/nursery/launch_local/requirements-synetune.txt``, and not in
``requirements.txt`` instead. The reason is that dependencies of the SageMaker
estimator for running the experiment locally is really the union of two such
files. First, ``requirements-synetune.txt`` for the Syne Tune dependencies,
and second, ``requirements.txt`` next to the training script. The remote
launching script is creating a ``requirements.txt`` file with this union in
``benchmarking/nursery/launch_local/``, which should not become part of the
repository.

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
``launch_remote.py`` or ``hpo_main.py`` as ``--random_seed`. If no master
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
