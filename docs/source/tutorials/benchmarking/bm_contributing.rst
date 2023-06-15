Contributing Your Benchmark
===========================

In order to increase its scope and usefulness, Syne Tune greatly welcomes the
contribution of new benchmarks, in particular in areas not yet well covered.
In a nutshell, contributing a benchmark is pretty similar to a code
contribution, but in this section, we provide some extra hints.

Contributing a Real Benchmark
-----------------------------

In principle, a real benchmark consists of a Python script which runs
evaluations, adhering to the conventions of Syne Tune. However, in order for
your benchmark to be useful for the community, here are some extra
requirements:

* The benchmark should not be excessively expensive to run
* If your benchmark involves training a machine learning model, the code should
  work with the dependencies of a
  `SageMaker framework <https://sagemaker.readthedocs.io/en/stable/frameworks/index.html>`__.
  You can specify extra dependencies, but they should be small. While Syne
  Tune (and SageMaker) supports Docker containers, Syne Tune is not hosting
  them. At present, we also do not accept ``Dockerfile`` script contributions,
  since we cannot maintain them.
* If your benchmark depends on data files, these must be hosted for public read
  access somewhere. Syne Tune cannot host data files, and will reject
  contributions with large files. If downloading and preprocessing the data for
  your benchmark takes too long, you may contribute an import script of a
  similar type to what is done in our :mod:`syne_tune.blackbox_repository`.

Let us have a look at the ``resnet_cifar10`` benchmark as example of what needs
to be done:

* `resnet_cifar10.py <training_scripts.html#resnet-18-trained-on-cifar-10>`__:
  The training script for your benchmark should be in a subdirectory of
  ``benchmarking/training_scripts/``. The same directory can contain a file
  ``requirements.txt`` with dependencies beyond the SageMaker framework you
  specify for your code. You are invited to study the code
  `resnet_cifar10.py <training_scripts.html#resnet-18-trained-on-cifar-10>`__
  in detail. Important points are:

  * Your script needs to report relevant metrics back to Syne Tune at the end
    of each epoch (or only once, at the end, if your script does not support
    multi-fidelity tuning), using an instance of :class:`~syne_tune.Reporter`.
  * We strongly recommend your script to support checkpointing, and the
    ``resnet_cifar10`` script is a good example for how to do this with
    ``PyTorch`` training scripts. If checkpointing is not supported, all
    pause-and-resume schedulers will run substantially slower than they really
    have to, because every resume operation requires them to train the model
    from scratch.

* :mod:`benchmarking.benchmark_definitions.resnet_cifar10`:
  You need to define some meta-data for your benchmark in
  :mod:`benchmarking.benchmark_definitions`. This should be a
  function returning a
  :class:`~syne_tune.experiments.benchmark_definitions.RealBenchmarkDefinition`
  object. Arguments should be a flag ``sagemaker_backend`` (``True`` for
  SageMaker backend experiment, ``False`` otherwise), and ``**kwargs``
  overwriting values in ``RealBenchmarkDefinition``. Hints:

  * ``framework`` should be one of the
    `SageMaker frameworks <https://sagemaker.readthedocs.io/en/stable/frameworks/index.html>`__.
    You should also specify ``framework_version`` and ``py_version`` in the
    ``estimator_kwargs`` dict.
  * ``config_space`` is the configuration space for your benchmark. Please
    make sure to
    `choose hyperparameter domains wisely <../../search_space.html>`__.
  * ``instance_type``, ``n_workers``: You need to specify a default instance
    type and number of workers for experiments running your benchmark. If in
    doubt, choose instances with the lowest costs. Currently, most of our GPU
    benchmarks use ``ml.g4dn.xlarge``, and CPU benchmarks use ``ml.c5.4xlarge``.
    Note that for experiments with the local backend (``sagemaker_backend=False``),
    the instance type must offer at least ``n_workers`` GPUs or CPU cores. For
    example, ``ml.g4dn.xlarge`` only has 1 GPU, while ``ml.g4dn.12xlarge``
    provides for ``n_workers=4``.
  * ``max_wallclock_time`` is a default value for the length of experiments
    running your benchmark, a value which depends on ``instance_type``,
    ``n_workers``. \* ``metric``, ``mode``, ``max_resource_attr``,
    ``resource_attr`` are required parameters for your benchmark, which are
    arguments to schedulers.

.. note::
   If you simply would like to run experiments with your own training code,
   it is not necessary for you to the :mod:`benchmarking` module at all. It
   just makes comparisons to other built-in benchmarks easier. See
   `this tutorial <../experimentation/README.html>`__ for more details.

Role of benchmarking/nursery/
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The best place to contribute a new benchmark, along with launcher scripts, is
to create a new package in :mod:`benchmarking.nursery`. This package contains:

* Training script and meta-data definition, as detailed above
* Launcher scripts, as detailed in the remainder of this tutorial
* Optionally, some scripts to visualize results

You are encouraged to run some experiments with your benchmark, involving a
number of baseline HPO methods, and submit results along with your pull
request.

Once your benchmark is in there, it may be used by the community. If others
find it useful, it can be graduated into
:mod:`benchmarking.benchmark_definitions`,
:mod:`benchmarking.training_scripts`, and :mod:`benchmarking.examples`.

We are looking forward to your
`pull request <https://github.com/awslabs/syne-tune/blob/main/CONTRIBUTING.md>`__.

Contributing a Tabulated Benchmark
----------------------------------

Syne Tune contains a blackbox repository :mod:`syne_tune.blackbox_repository`
for maintaining and serving tabulated and surrogate benchmarks, as well as a
simulator backend (:mod:`syne_tune.backend.simulator_backend`), which
simulates training evaluations from a blackbox. The simulator backend can be
used with any Syne Tune scheduler, and experiment runs are very close to what
would be obtained by running training for real. Since time is simulated as well,
not only are experiments very cheap to run (on basic CPU hardware), they also
finish many times faster than real time. An overview is given
`here <../multifidelity/mf_setup.html>`__.

If you have the data for a tabulated benchmark, we strongly encourage you to
`contribute an import script to Syne Tune <https://github.com/awslabs/syne-tune/blob/main/CONTRIBUTING.md>`__.
Examples for such scripts are
:mod:`syne_tune.blackbox_repository.conversion_scripts.scripts.fcnet_import`,
:mod:`syne_tune.blackbox_repository.conversion_scripts.scripts.nasbench201_import`,
:mod:`syne_tune.blackbox_repository.conversion_scripts.scripts.pd1_import`,
:mod:`syne_tune.blackbox_repository.conversion_scripts.scripts.yahpo_import`,
:mod:`syne_tune.blackbox_repository.conversion_scripts.scripts.lcbench.lcbench`.
See also
`FAQ <../../faq.html#how-can-i-add-a-new-tabular-or-surrogate-benchmark>`__.
