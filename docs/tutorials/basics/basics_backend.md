# Basics of Syne Tune: SageMaker Back-end


[[Previous Section]](basics_promotion.md)


## Limitations of the Local Back-end

We have been using the local back-end
[LocalBackend](../../../syne_tune/backend/local_backend.py) in this tutorial so
far. Due to its simplicity and very low overheads for starting, stopping, or
resuming trials, this is the preferred choice for getting started. But with
models and datasets getting larger, some disadvantages become apparent:
* All concurrent training jobs (as well as the tuning algorithm itself) are run
  as subprocesses on the same instance. This limits the number of workers by
  what is offered by the instance type. You can set `n_workers` to any value
  you like, but what you really get depends on available resources. If you want
  4 GPU workers, your instance types needs to have at least 4 GPUs, and each
  training job can use only one of them.
* It is hard to encapsulate dependencies of your training code. You need to
  specify them explicitly, and they need to be compatible with the Syne Tune
  dependencies. You cannot use Docker images.
* You may be used to work with SageMaker frameworks, or even specialized setups
  such as distributed training. In such cases, it is hard to get tuning to work
  with the local back-end.

At present, Syne Tune offers the SageMaker back-end as alternative to the local
one. Support of further back-ends is planned, but not available yet.


## Launcher Script for SageMaker Back-end

Before continuing, please make sure you are able to start SageMaker training
jobs from wherever you are launching Syne Tune experiments. If you cannot get
this working on your local machine, you may start a SageMaker notebook instance
and install Syne Tune there.

If you are used to the SageMaker Python API, and in particular to SageMaker
estimators, the following will come natural.
[launch_sagemaker_backend.py](scripts/launch_sagemaker_backend.py) is a
launcher script for MOBSTER (stopping-type), using the SageMaker instead of
the local back-end.
* [1] The only difference is that `backend`, feeding into the `Tuner`, is
  different. Instead of the code `entry_point` only, you need to specify a
  [SageMaker estimator](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html)
  to be used for executing your code as a SageMaker training job. Our running
  example is in PyTorch, so we use the `PyTorch` estimator. Except for
  `entry_point`, arguments depend on the estimator type. Some of the most
  relevant ones are `instance_type`, `instance_count` (one training job can use
  several instances), and `role`. The latter is your execution role for the
  training job: with `get_execution_role()`, you use the default one associated
  with your AWS account. Finally, `max_run` limits the time of a training job,
  make sure this is not much shorter than your experiment time.
  Apart from `sm_estimator`, you can also specify the names of metrics reported
  to Syne Tune in `metrics_names`, these will be displayed in the dashboard of
  the training job.
* It is important to specify dependencies of the training script not contained
  in the SageMaker framework. For source dependencies, you use the
  `dependencies` parameter. While `syne_tune` is automatically added, you need
  to add `benchmarking` in order to use support code from there. Extra PyPI
  dependencies can be listed in a file `requirements.txt` stored in the same
  directory as your script. Our running example needs `filelock`, for example.

This launcher script shows how to run training code whose dependencies are
captured by a SageMaker framework (`PyTorch` in this case), and extra ones
are specified by a `dependencies.txt` file. However, you may prefer to use a
Docker image for your training code.
[launch_height_sagemaker_custom_image.py](examples/launch_height_sagemaker_custom_image.py)
is an example for how to use such a custom image.


## Properties of the SageMaker Back-end

Our running example is not a good candidate to use the SageMaker back-end for.
It is small, cheap and easy to run, and for such tuning problems, the local
back-end is the preferred choice. The SageMaker back-end starts a new training
job for every trial. While this offers great flexibility (you can select a
SageMaker framework, instance type, instance count), it comes with a substantial
start-up overhead (often almost two minutes). If training jobs are much longer
than that, as is often the case with `FIFOScheduler`, these over-heads may not
matter much.

However, for early stopping schedulers like `HyperbandScheduler`, most trials
do not run for long, but are stopped or paused after one or a few epochs. For such
algorithms start-up overheads can matter a lot more. On the other hand, they 
also profit from increased parallelization.


In the [final section](basics_outlook.md), we provide an outlook to further
topics.
