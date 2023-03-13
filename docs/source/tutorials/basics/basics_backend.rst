SageMaker Backend
==================

Limitations of the Local Backend
---------------------------------

We have been using the local backend :class:`~syne_tune.backend.LocalBackend`
in this tutorial so far. Due to its simplicity and very low overheads for
starting, stopping, or resuming trials, this is the preferred choice for
getting started. But with models and datasets getting larger, some
disadvantages become apparent:

* All concurrent training jobs (as well as the tuning algorithm itself) are
  run as subprocesses on the same instance. This limits the number of workers
  by what is offered by the instance type. You can set ``n_workers`` to any
  value you like, but what you really get depends on available resources. If
  you want 4 GPU workers, your instance types needs to have at least 4 GPUs,
  and each training job can use only one of them.
* It is hard to encapsulate dependencies of your training code. You need to
  specify them explicitly, and they need to be compatible with the Syne Tune
  dependencies. You cannot use Docker images.
* You may be used to work with SageMaker frameworks, or even specialized setups
  such as distributed training. In such cases, it is hard to get tuning to work
  with the local backend.


Launcher Script for SageMaker Backend
--------------------------------------

Syne Tune offers the SageMaker backend
:class:`~syne_tune.backend.SageMakerBackend` as alternative to the local one.
Using it requires some preparation, as is detailed
`here <../../faq.html#how-can-i-run-on-aws-and-sagemaker>`__.

Recall our
`launcher script <basics_randomsearch.html#launcher-script-for-random-search>`__.
In order to use the SageMaker backend, we need to create ``trial_backend``
differently:

.. code-block:: python

   trial_backend = SageMakerBackend(
       # we tune a PyTorch Framework from Sagemaker
       sm_estimator=PyTorch(
           entry_point=entry_point.name,
           source_dir=str(entry_point.parent),
           instance_type="ml.c5.4xlarge",
           instance_count=1,
           role=get_execution_role(),
           dependencies=[str(repository_root_path() / "benchmarking")],
           max_run=int(1.05 * args.max_wallclock_time),
           framework_version="1.7.1",
           py_version="py3",
           disable_profiler=True,
           debugger_hook_config=False,
           sagemaker_session=default_sagemaker_session(),
       ),
       metrics_names=[metric],
   )

In essence, the :class:`~syne_tune.backend.SageMakerBackend` is parameterized
with a SageMaker estimator, which executes the training script. In our example,
we use the ``PyTorch`` SageMaker framework as a pre-built container for the
dependencies our training scripts requires. However, any other type of
`SageMaker estimator <https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html>`__
can be used here just as well. Finally, if you include any of the metrics reported
by your training script in ``metrics_names``, their values are visualized in the
dashboard for the SageMaker training job.

If your training script requires additional dependencies not contained in the
chosen SageMaker framework, you can specify those in a ``requirements.txt``
file in the same directory as your training script (i.e., in the ``source_dir``
of the SageMaker estimator). In our example, this file needs to contain the
``filelock`` dependence.

.. note::
   This simple example avoids complications about writing results to S3 in
   a unified manner, or using special features of SageMaker which can speed
   up tuning substantially. For more information about the SageMaker backend,
   please consider `this tutorial <../benchmarking/bm_sagemaker.html>`__.
