Why should I use Syne Tune?
===========================

Hyperparameter Optimization (HPO) has been an important problem for many years,
and a variety of commercial and open-source tools are available to help
practitioners run HPO efficiently. Notable examples for open source
tools are `Ray Tune <https://docs.ray.io/en/latest/tune/index.html>`__ and
`Optuna <https://optuna.readthedocs.io/en/stable/>`__. Here are some reasons
why you may prefer Syne Tune over these alternatives:

* **Lightweight and platform-agnostic**: Syne Tune is designed to work with
  different execution backends, so you are not locked into a particular
  distributed system architecture. Syne Tune runs with minimal dependencies.
* **Wide range of modalities**: Syne Tune supports multi-fidelity HPO,
  constrained HPO, multi-objective HPO, transfer tuning, cost-aware HPO,
  population based training.
* **Simple, modular design**: Rather than wrapping all sorts of other HPO
  frameworks, Syne Tune provides simple APIs and scheduler templates, which can
  easily be `extended to your specific needs <tutorials/developer/README.html>`__.
* **Industry-strength Bayesian optimization**: Syne Tune has special support
  for `Gaussian process based Bayesian optimization <tutorials/basics/basics_bayesopt.html>`__.
  The same code powers modalities like multi-fidelity HPO, constrained HPO, or
  cost-aware HPO, having been tried and tested for several years.
* **Special support for researchers**: Syne Tune allows for rapid development
  and comparison between different tuning algorithms. Its
  `blackbox repository and simulator backend <tutorials/multifidelity/mf_setup.html>`__
  run realistic simulations of experiments many times faster than real time.
  `Benchmarking <tutorials/benchmarking/README.html>`__ is simple and efficient.

If you are an AWS customer, there are additional good reasons to use Syne Tune
over the alternatives:

* If you use AWS services or SageMaker frameworks day to day, Syne Tune works
  out of the box and fits into your normal workflow.
* Syne Tune is developed in collaboration with the team behind the
  `Automatic Model Tuning <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html>`__
  service.

What are the different installation options supported?
=======================================================

To install Syne Tune with minimal dependencies from pip, you can simply do:

.. code:: bash

   pip install 'syne-tune'

If you want in addition to install our own Gaussian process based optimizers,
Ray Tune or Bore optimizer, you can run ``pip install 'syne-tune[X]'`` where
``X`` can be:

* ``gpsearchers``: For built-in Gaussian process based optimizers (such as
  :class:`~syne_tune.optimizer.baselines.BayesianOptimization`,
  :class:`~syne_tune.optimizer.baselines.MOBSTER`, or
  :class:`~syne_tune.optimizer.baselines.HyperTune`)
* ``aws``: AWS SageMaker dependencies. These are required for
  `remote launching <#i-dont-want-to-wait-how-can-i-launch-the-tuning-on-a-remote-machine>`__
  or for the :class:`~syne_tune.backend.SageMakerBackend`
* ``raytune``: For Ray Tune optimizers (see
  :class:`~syne_tune.optimizer.schedulers.RayTuneScheduler`), installs all Ray
  Tune dependencies
* ``benchmarks``: For installing dependencies required to run all benchmarks
  locally (not needed for remote launching or
  :class:`~syne_tune.backend.SageMakerBackend`)
* ``blackbox-repository``: Blackbox repository for simulated tuning
* ``yahpo``: YAHPO Gym surrogate blackboxes
* ``kde``: For BOHB (such as :class:`~syne_tune.optimizer.baselines.SyncBOHB`,
  or :class:`~syne_tune.optimizer.schedulers.FIFOScheduler` or
  :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` with
  ``searcher="kde"``)
* ``botorch``: Bayesian optimization from BoTorch (see
  :class:`~syne_tune.optimizer.schedulers.searchers.botorch.BoTorchSearcher`)
* ``dev``: For developers who would like to extend Syne Tune
* ``extra``: For installing all the above
* ``bore``: For Bore optimizer (see :class:`~syne_tune.optimizer.baselines.BORE`)

For instance, ``pip install 'syne-tune[gpsearchers]'`` will install Syne
Tune along with many built-in Gaussian process optimizers.

To install the latest version from git, run the following:

.. code:: bash

   pip install git+https://github.com/awslabs/syne-tune.git

For local development, we recommend using the following setup which will
enable you to easily test your changes:

.. code:: bash

   git clone https://github.com/awslabs/syne-tune.git
   cd syne-tune
   python3 -m venv st_venv
   . st_venv/bin/activate
   pip install --upgrade pip
   pip install -e '.[extra]'

This installs everything in a virtual environment ``st_venv``. Remember to
activate this environment before working with Syne Tune. We also recommend
building the virtual environment from scratch now and then, in particular when
you pull a new release, as dependencies may have changed.

How can I run on AWS and SageMaker?
===================================

If you want to launch experiments or training jobs on SageMaker rather than on
your local machine, you will need access to AWS and SageMaker on your machine.
Make sure that:

* ``awscli`` is installed (see
  `this link <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html>`__)
* AWS credentials have been set properly (see
  `this link <https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html>`__).
* The necessary SageMaker role has been created (see
  `this page <https://docs.aws.amazon.com/glue/latest/dg/create-an-iam-role-sagemaker-notebook.html>`__
  for instructions. If you’ve created a SageMaker notebook in the past, this
  role should already have been created for you).

The following command should run without error if your credentials are available:

.. code:: bash

   python -c "import boto3; print(boto3.client('sagemaker').list_training_jobs(MaxResults=1))"

You can also run the following example that evaluates trials on SageMaker to
test your setup.

.. code:: bash

   python examples/launch_height_sagemaker.py

What are the metrics reported by default when calling the ``Reporter``?
=======================================================================

Whenever you call the reporter to log a result, the worker time-stamp, the
worker time since the creation of the reporter and the number of times the
reporter was called are logged under the fields
:const:`~syne_tune.constants.ST_WORKER_TIMESTAMP`,
:const:`~syne_tune.constants.ST_WORKER_TIME`, and
:const:`~syne_tune.constants.ST_WORKER_ITER`. In addition, when running on
SageMaker, a dollar-cost estimate is logged under the field
:const:`~syne_tune.constants.ST_WORKER_COST`.

To see this behavior, you can simply call the reporter to see those
metrics:

.. code:: python

   from syne_tune.report import Reporter
   reporter = Reporter()
   for step in range(3):
      reporter(step=step, metric=float(step) / 3)
      
   # [tune-metric]: {"step": 0, "metric": 0.0, "st_worker_timestamp": 1644311849.6071281, "st_worker_time": 0.0001048670000045604, "st_worker_iter": 0}
   # [tune-metric]: {"step": 1, "metric": 0.3333333333333333, "st_worker_timestamp": 1644311849.6071832, "st_worker_time": 0.00015910100000837701, "st_worker_iter": 1}
   # [tune-metric]: {"step": 2, "metric": 0.6666666666666666, "st_worker_timestamp": 1644311849.60733, "st_worker_time": 0.00030723599996917983, "st_worker_iter": 2}

How can I utilize multiple GPUs?
================================

To utilize multiple GPUs you can either use the backend
:class:`~syne_tune.backend.LocalBackend`, which will run on the GPU available
in a local machine. You can also run on a remote AWS instance with multiple GPUs
using the local backend and the remote launcher, see
`here <#i-dont-want-to-wait-how-can-i-launch-the-tuning-on-a-remote-machine>`__,
or run with the :class:`~syne_tune.backend.SageMakerBackend` which spins-up one
training job per trial.

When evaluating trials on a local machine with
:class:`~syne_tune.backend.LocalBackend`, by default each trial is allocated to
the least occupied GPU by setting ``CUDA_VISIBLE_DEVICES`` environment
variable.

What is the default mode when performing optimization?
======================================================

The default mode is ``"min"`` when performing optimization, so the target metric
is minimized. The mode can be configured when instantiating a scheduler.

How are trials evaluated on a local machine?
============================================

When trials are executed locally (e.g., when
:class:`~syne_tune.backend.LocalBackend` is used), each trial is evaluated as a
different sub-process. As such the number of concurrent configurations evaluated
at the same time (set by ``n_workers`` when creating the
:class:`~syne_tune.Tuner`) should account for the capacity of the machine where
the trials are executed.

Where can I find the output of the tuning?
==========================================

When running locally, the output of the tuning is saved under
``~/syne-tune/{tuner-name}/`` by default. When running remotely on SageMaker,
the output of the tuning is saved under ``/opt/ml/checkpoints/`` by default and
the tuning output is synced regularly to
``s3://{sagemaker-default-bucket}/syne-tune/{tuner-name}/``.

How can I change the default output folder where tuning results are stored?
===========================================================================

To change the path where tuning results are written, you can set the
environment variable ``SYNETUNE_FOLDER`` to the folder that you want.

For instance, the following runs a tuning where results tuning files are
written under ``~/new-syne-tune-folder``:

.. code:: bash

   export SYNETUNE_FOLDER="~/new-syne-tune-folder"
   python examples/launch_height_baselines.py

You can also do the following for instance to permanently change the output
folder of Syne Tune:

.. code:: bash

   echo 'export SYNETUNE_FOLDER="~/new-syne-tune-folder"' >> ~/.bashrc && source ~/.bashrc

What does the output of the tuning contain?
===========================================

Syne Tune stores the following files ``metadata.json``, ``results.csv.zip``,
and ``tuner.dill``, which are respectively metadata of the tuning job, results
obtained at each time-step, and state of the tuner. If you create the
:class:`~syne_tune.Tuner` with ``save_tuner=False``, the ``tuner.dill`` file is
not written. The content of ``results.csv.zip``
`can be customized <#how-can-i-write-extra-results-for-an-experiment>`__.

How can I enable trial checkpointing?
=====================================

Since trials may be paused and resumed (either by schedulers or when using
spot-instances), the user may checkpoint intermediate results to avoid starting
computation from scratch. Model outputs and checkpoints must be written into a
specific local path given by the command line argument
:const:`~syne_tune.constants.ST_CHECKPOINT_DIR`. Saving/loading model checkpoint
from this directory enables to save/load the state when the job is
stopped/resumed (setting the folder correctly and uniquely per trial is the
responsibility of the backend). Here is an example of a tuning script with
checkpointing enabled:

.. literalinclude:: ../../examples/training_scripts/checkpoint_example/checkpoint_example.py
   :name: checkpoint_example_script
   :caption: examples/training_scripts/checkpoint_example/checkpoint_example.py
   :start-after: # permissions and limitations under the License.

When using the SageMaker backend, we use the
`SageMaker checkpoint mechanism <https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html>`__
under the hood to sync local checkpoints to S3. Checkpoints are synced to
``s3://{sagemaker-default-bucket}/syne-tune/{tuner-name}/{trial-id}/``,
where ``sagemaker-default-bucket`` is the default bucket for SageMaker. A complete
example is given by
`examples/launch_height_sagemaker_checkpoints.py <examples.html#sageMaker-backend-and-checkpointing>`__.

The same mechanism is used to regularly write the
`tuning results to S3 during remote tuning <#where-can-i-find-the-output-of-the-tuning>`__.
However, during remote tuning with the *local backend*, we do not want
checkpoints to be synced to S3, since they are only required temporarily on the
same instance. Syncing them to S3 would be costly and error-prone, because the
SageMaker mechanism is not intended to work with different processes writing to
and reading from the sync directory concurrently. In this case, we can switch
off syncing checkpoints to S3 (but not tuning results!) by setting
``trial_backend_path=backend_path_not_synced_to_s3()`` when creating the
:class:`~syne_tune.Tuner` object. An example is
`fine_tuning_transformer_glue/hpo_main.py <benchmarking/fine_tuning_transformer_glue.html>`__.
It is also supported by default in the
`benchmarking framework <tutorials/benchmarking/README.html>`__ and in
:class:`~syne_tune.remote.RemoteLauncher`.

There are some convenience functions which help you to implement checkpointing
for your training script. Have a look at
`resnet_cifar10.py <training_scripts.html#resnet-18-trained-on-cifar-10>`__:

* Checkpoints have to be written at the end of certain epochs (namely
  those after which the scheduler may pause the trial). This is dealt with
  by ``checkpoint_model_at_rung_level(config, save_model_fn, epoch)``.
  Here, ``epoch`` is the current epoch, allowing the function to decide
  whether to checkpoint or not. ``save_model_fn`` stores the current
  mutable state along with ``epoch`` to a local path (see below). Finally,
  ``config`` contains arguments provided by the scheduler (see below).
* Before the training loop starts (and optionally), the mutable state to
  start from has to be loaded from a checkpoint. This is done by
  ``resume_from_checkpointed_model(config, load_model_fn)``. If the
  checkpoint has been loaded successfully, the training loop may start with
  epoch ``resume_from + 1`` instead of ``1``. Here, ``load_model_fn`` loads
  the mutable state from a checkpoint in a local path, returning its
  ``epoch`` value if successful, which is returned as ``resume_from``.

In general, ``load_model_fn`` and ``save_model_fn`` have to be provided as part
of the script. For most PyTorch models, you can use
``pytorch_load_save_functions`` to this end. In general, you will want to
include the model, the optimizer, and the learning rate scheduler.

Finally, the scheduler provides additional information about checkpointing in
``config`` (most importantly, the path in
:const:`~syne_tune.constants.ST_CHECKPOINT_DIR`). You don’t have to worry about
this: ``add_checkpointing_to_argparse(parser)`` adds corresponding arguments to
the parser.

How can I retrieve the best checkpoint obtained after tuning?
=============================================================

You can take a look at this example
`examples/launch_checkpoint_example.py <examples.html#retrieving-the-best-checkpoint>`__
which shows how to retrieve the best checkpoint obtained after tuning an XGBoost model.

Which schedulers make use of checkpointing?
===========================================

Checkpointing means storing the state of a trial (i.e., model parameters,
optimizer or learning rate scheduler parameters), so that it can be paused and
potentially resumed at a later point in time, without having to start training
from scratch. The following schedulers make use of checkpointing:

* Promotion-based asynchronous Hyperband:
  :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` with
  ``type="promotion"`` or ``type="dyhpo"``, as well as other asynchronous multi-fidelity schedulers.
  The code runs without checkpointing, but in this case, any trial which is
  resumed is started from scratch. For example, if a trial was paused after 9
  epochs of training and is resumed later, training starts from scratch
  and the first 9 epochs are wasted effort. Moreover, extra variance is
  introduced by starting from scratch, since weights may be initialized
  differently. It is not recommended running promotion-based Hyperband
  without checkpointing.
* Population-based training (PBT):
  :class:`~syne_tune.optimizer.schedulers.PopulationBasedTraining` does not
  work without checkpointing.
* Synchronous Hyperband:
  :class:`~syne_tune.optimizer.schedulers.synchronous.SynchronousGeometricHyperbandScheduler`,
  as well as other synchronous multi-fidelity schedulers. This code runs
  without checkpointing, but wastes effort in the same sense as
  promotion-based asynchronous Hyperband

Checkpoints are filling up my disk. What can I do?
==================================================

When tuning large models, checkpoints can be large, and with the local backend,
these checkpoints are stored locally. With multi-fidelity methods, many trials
may be started, and keeping all checkpoints (which is the default) may exceed
the available disk space.

If the trial backend :class:`~syne_tune.backend.trial_backend.TrialBackend` is
created with ``delete_checkpoints=True``, Syne Tune removes the checkpoint of a
trial once it is stopped or completes. All remaining checkpoints are removed at
the end of the experiment. Moreover, a number of schedulers support early
checkpoint removal for paused trials when they cannot be resumed anymore.

For promotion-based asynchronous multi-fidelity schedulers (
`ASHA <tutorials/multifidelity/mf_asha.html>`__,
`MOBSTER <tutorials/multifidelity/mf_async_model.html#asynchronous-mobster>`__,
`HyperTune <tutorials/multifidelity/mf_async_model.html#hyper-tune>`__), any
paused trial can in principle be resumed in the future, and
`delete_checkpoints=True`` alone does not remove checkpoints. In this case,
you can activate speculative early checkpoint removal, by passing
``early_checkpoint_removal_kwargs`` when creating
:class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` (or
:class:`~syne_tune.optimizer.baselines.ASHA`,
:class:`~syne_tune.optimizer.baselines.MOBSTER`,
:class:`~syne_tune.optimizer.baselines.HyperTune`). This is a ``kwargs``
dictionary with the following arguments:

* ``max_num_checkpoints``: This is mandatory. Maximum number of trials with
  checkpoints being retained. Once more than this number of trials with
  checkpoints are present, checkpoints are removed selectively. This number must
  be larger than the number of workers, since running trials will always write
  checkpoints.
* ``approx_steps``: Positive integer. The computation of the ranking score is
  a step-wise approximation, which gets more accurate for larger ``approx_steps``.
  However, this computation scales cubically in ``approx_steps``. The default
  is 25, which may be sufficient in most cases, but if you need to keep the
  number of checkpoints quite small, you may want to tune this parameter.
* ``max_wallclock_time``: Maximum time in seconds the experiment is run for. This
  is the same as passed to :class:`~syne_tune.StoppingCriterion`, and if you use
  an instance of this as ``stop_criterion`` passed to :class:`~syne_tune.Tuner`,
  the value is taken from there. Speculative checkpoint removal can only be used
  if the stopping criterion includes ``max_wallclock_time``.
* ``prior_beta_mean``: The method depends on the probability of the event
  that a trial arriving at a rung ranks better than a random paused trial
  with checkpoint at this rung. These probabilities are estimated for each
  rung, but we need some initial guess. You are most likely fine with the
  default. A value :math:`< 1/2` is recommended.
* ``prior_beta_size``: See also ``prior_beta_mean``. The initial guess is
  a Beta prior, defined in terms of mean and effective sample size (here).
  The smaller this positive number, the weaker the effect of the initial
  guess. You are most likely fine with the default.
* ``min_data_at_rung``: Also related to the estimators mentioned with
  ``prior_beta_mean``. You are most likely fine with the default.

A complete example is
`examples/launch_fashionmnist_checkpoint_removal.py <examples.html#speculative-early-checkpoint-removal>`__.
For details on speculative checkpoint removal, look at
:class:`~syne_tune.callbacks.hyperband_remove_checkpoints_callback.HyperbandRemoveCheckpointsCallback`.

Is the tuner checkpointed?
==========================

Yes. When performing the tuning, the tuner state is regularly saved on the
experiment path under ``tuner.dill`` (every 10 seconds, which can be configured
with ``results_update_interval`` when creating the :class:`~syne_tune.Tuner`).
This allows to use spot-instances when running a tuning remotely with the remote
launcher. It also allows to resume a past experiment or analyse the state of
scheduler at any point.

Where can I find the output of my trials?
=========================================

When running :class:`~syne_tune.backend.LocalBackend` locally, results of trials
are saved under ``~/syne-tune/{tuner-name}/{trial-id}/`` and contains the
following files:

-  ``config.json``: configuration that is being evaluated in the trial
-  ``std.err``: standard error
-  ``std.out``: standard output

In addition all checkpointing files used by a training script such as
intermediate model checkpoint will also be located there. This is exemplified
in the following example:

.. code-block:: bash

   tree ~/syne-tune/train-height-2022-01-12-11-08-40-971/
   ~/syne-tune/train-height-2022-01-12-11-08-40-971/
   ├── 0
   │   ├── config.json
   │   ├── std.err
   │   ├── std.out
   │   └── stop
   ├── 1
   │   ├── config.json
   │   ├── std.err
   │   ├── std.out
   │   └── stop
   ├── 2
   │   ├── config.json
   │   ├── std.err
   │   ├── std.out
   │   └── stop
   ├── 3
   │   ├── config.json
   │   ├── std.err
   │   ├── std.out
   │   └── stop
   ├── metadata.json
   ├── results.csv.zip
   └── tuner.dill

When running tuning remotely with the remote launcher, only ``config.json``,
``metadata.json``, ``results.csv.zip`` and ``tuner.dill`` are synced with S3
unless ``store_logs_localbackend=True`` when creating :class:`~syne_tune.Tuner`,
in which case the trial logs and informations are also persisted.

How can I plot the results of a tuning?
=======================================

Some basic plots can be obtained via
:class:`~syne_tune.experiments.ExperimentResult`. An example is given in
`examples/launch_plot_results.py <examples.html#plot-results-of-tuning-experiment>`__.

How can I specify additional tuning metadata?
=============================================

By default, Syne Tune stores the time, the names and modes of the metric being
tuned, the name of the entrypoint, the name backend and the scheduler name. You
can also add custom metadata to your tuning job by setting ``metadata`` in
:class:`~syne_tune.Tuner` as follow:

.. code:: python

   from syne_tune import Tuner

   tuner = Tuner(
       ...
       tuner_name="plot-results-demo",
       metadata={"tag": "special-tag", "user": "alice"},
   )

All Syne Tune and user metadata are saved when the tuner starts under
``metadata.json``.

How do I append additional information to the results which are stored?
=======================================================================

Results are processed and stored by callbacks passed to
:class:`~syne_tune.Tuner`, in particular see
:class:`~syne_tune.tuner_callback.StoreResultsCallback`. In order to add more
information, you can inherit from this class. An example is given in
:class:`~syne_tune.optimizer.schedulers.searchers.searcher_callback.StoreResultsAndModelParamsCallback`.

If you run experiments with tabulated benchmarks using the
:class:`~syne_tune.blackbox_repository.BlackboxRepositoryBackend`, as demonstrated in
`launch_nasbench201_simulated.py <examples.html#launch-hpo-experiment-with-simulator-backend>`__,
results are stored by
:class:`~syne_tune.backend.simulator_backend.simulator_callback.SimulatorCallback`
instead, and you need to inherit from this class. An example is given in
:class:`~syne_tune.optimizer.schedulers.searchers.searcher_callback.SimulatorAndModelParamsCallback`.

I don’t want to wait, how can I launch the tuning on a remote machine?
======================================================================

Remote launching of experiments has a number of advantages:

* The machine you are working on is not blocked
* You can launch many experiments in parallel
* You can launch experiments with any instance type you like, without having to
  provision them yourselves. For GPU instances, you do not have to worry about
  setting up CUDA, etc.

You can use the remote launcher to launch an experiment on a remote machine.
The remote launcher supports both :class:`~syne_tune.backend.LocalBackend` and
:class:`~syne_tune.backend.SageMakerBackend`. In the former case, multiple
trials will be evaluated on the remote machine (one use-case being to use a
beefy machine), in the latter case trials will be evaluated as separate
SageMaker training jobs. An example for running the remote launcher is
given in
`launch_height_sagemaker_remotely.py <examples.html#launch-experiments-remotely-on-sagemaker>`__.

Remote launching for benchmarking (i.e., running many remote experiments
in order to compare multiple methods) is detailed in
`this tutorial <tutorials/benchmarking/README.html>`__.

How can I run many experiments in parallel?
===========================================

You can remotely launch any number of experiments, which will then run
in parallel, as detailed in
`this tutorial <tutorials/benchmarking/README.html>`__, see also these examples:

* Local backend:
  `benchmarking/nursery/launch_local/ <benchmarking/launch_local.html>`__
* Simulator backend:
  `benchmarking/nursery/benchmark_dehb/ <benchmarking/benchmark_dehb.html>`__
* SageMaker backend:
  `benchmarking/nursery/launch_sagemaker/ <benchmarking/launch_sagemaker.html>`__

.. note::
   In order to use the *benchmarking* framework, you need to have
   `installed Syne Tune from source <getting_started.html#installation>`__.

How can I access results after tuning remotely?
===============================================

You can either call :func:`~syne_tune.experiments.load_experiment`, which will
download files from S3 if the experiment is not found locally. You can also
sync directly files from S3 under ``~/syne-tune/`` folder in batch for instance
by running:

.. code:: bash

   aws s3 sync s3://{sagemaker-default-bucket}/syne-tune/{tuner-name}/ ~/syne-tune/  --include "*"  --exclude "*tuner.dill"

To get all results without the tuner state (you can omit the ``include``
and ``exclude`` if you also want to include the tuner state).

How can I specify dependencies to remote launcher or when using the SageMaker backend?
======================================================================================

When you run remote code, you often need to install packages
(e.g., ``scipy``) or have custom code available.

* To install packages, you can add a file ``requirements.txt`` in the
  same folder as your endpoint script. All those packages will be
  installed by SageMaker when docker container starts.
* To include custom code (for instance a library that you are working
  on), you can set the parameter ``dependencies`` on the remote
  launcher or on a SageMaker framework to a list of folders. The
  folders indicated will be compressed, sent to S3 and added to the
  python path when the container starts. More details are given in
  `this tutorial <tutorials/benchmarking/bm_sagemaker.html>`__.

How can I benchmark different methods?
======================================

The most flexible way to do so is to write a custom launcher script, as detailed
in `this tutorial <tutorials/benchmarking/README.html>`__, see also these examples:

* Local backend:
  `benchmarking/nursery/launch_local/ <benchmarking/launch_local.html>`__
* Simulator backend:
  `benchmarking/nursery/benchmark_dehb/ <benchmarking/benchmark_dehb.html>`__
* SageMaker backend:
  `benchmarking/nursery/launch_sagemaker/ <benchmarking/launch_sagemaker.html>`__
* Fine-tuning transformers:
  `benchmarking/nursery/fine_tuning_transformer_glue/ <benchmarking/fine_tuning_transformer_glue.html>`__
* Hyper-Tune:
  `benchmarking/nursery/benchmark_hypertune/ <benchmarking/benchmark_hypertune.html>`__

.. note::
   In order to use the *benchmarking* framework, you need to have
   `installed Syne Tune from source <getting_started.html#installation>`__.

What different schedulers do you support? What are the main differences between them?
=====================================================================================

A succinct overview of supported schedulers is provided
`here <getting_started.html#supported-hpo-methods>`__.

Most methods can be accessed with short names by from
:mod:`syne_tune.optimizer.baselines`, which is the best place to start.

We refer to HPO algorithms as *schedulers*. A scheduler decides which
configurations to assign to new trials, but also when to stop a running
or resume a paused trial. Some schedulers delegate the first decision to
a *searcher*. The most important differences between schedulers in the
single-objective case are:

* Does the scheduler stop trials early or pause and resume trials
  (:class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`) or not
  (:class:`~syne_tune.optimizer.schedulers.FIFOScheduler`). The former
  requires a resource dimension (e.g., number of epochs; size of
  training set) and slightly more elaborate reporting (e.g., evaluation
  after every epoch), but can outperform the latter by a large margin.
* Does the searcher suggest new configurations by uniform random
  sampling (``searcher="random"``) or by sequential model-based
  decision-making (``searcher="bayesopt"``, ``searcher="kde"``,
  ``searcher="hypertune"``, ``searcher="botorch"``, ``searcher="dyhpo"``).
  The latter can be more expensive if a lot of trials are run, but can also
  be more sample-efficient.

An overview of this landscape is given `here <schedulers.html>`__.

Here is a
`tutorial for multi-fidelity schedulers <tutorials/multifidelity/README.html>`__.
Further schedulers provided by Syne Tune include:

* `Population based training (PBT) <examples.html#population-based-training-pbt>`__
* `Multi-objective asynchronous successive halving (MOASHA) <examples.html#multi-objective-asynchronous-successive-halving-moasha>`__
* `Constrained Bayesian optimization <examples.html#constrained-bayesian-optimization>`__
* Bayesian optimization by density-ratio estimation:
  :class:`~syne_tune.optimizer.baselines.BORE`
* Regularized evolution: :class:`~syne_tune.optimizer.baselines.REA`
* Median stopping rule:
  :class:`~syne_tune.optimizer.schedulers.median_stopping_rule.MedianStoppingRule`
* `Synchronous Hyperband <tutorials/multifidelity/mf_syncsh.html>`__
* `Differential Evolution Hyperband (DEHB) <tutorials/multifidelity/mf_sync_model.html#differential-evolution-hyperband>`__
* `Hyper-Tune <tutorials/multifidelity/mf_async_model.html#hyper-tune>`__
* `DyHPO <tutorials/multifidelity/mf_async_model.html#dyhpo>`__
* `Transfer learning schedulers <examples.html#transfer-tuning-on-nasbench-201>`__
* `Wrappers for Ray Tune schedulers <examples.html#launch-hpo-experiment-with-ray-tune-scheduler>`__

How do I define the search space?
=================================

While the training script defines the function to be optimized, some
care needs to be taken to define the search space for the hyperparameter
optimization problem. This being a global optimization problem without
gradients easily available, it is most important to reduce the number of
parameters. Some advice is given `here <search_space.html>`__.

A powerful approach is to run experiments in parallel. Namely, split
your hyperparameters into groups A, B, such that HPO over B is
tractable. Draw a set of N configurations from A at random, then start N
HPO experiments in parallel, where in each of them the search space is
over B only, while the parameters in A are fixed. Syne Tune supports
massively parallel experimentation, see
`this tutorial <tutorials/benchmarking/README.html>`__.

How do I set arguments of multi-fidelity schedulers?
====================================================

When running schedulers like :class:`~syne_tune.optimizer.baselines.ASHA`,
:class:`~syne_tune.optimizer.baselines.MOBSTER`,
:class:`~syne_tune.optimizer.baselines.HyperTune`,
:class:`~syne_tune.optimizer.baselines.SyncHyperband`,
or :class:`~syne_tune.optimizer.baselines.DEHB`, there are mandatory parameters
``resource_attr``, ``max_resource_attr``, ``max_t``, ``max_resource_value``.
What are they for?

Full details are given in this
`tutorial <tutorials/multifidelity/README.html>`__. Multi-fidelity HPO needs
metric values to be reported at regular intervals during training, for example
after every epoch, or for successively larger training datasets. These
reports are indexed by a *resource value*, which is a positive integer (for
example, the number of epochs already trained).

* ``resource_attr`` is the name of the resource attribute in the dictionary
  reported by the training script. For example, the script may report
  :code:`report(epoch=5, mean_loss=0.125)` at the end of the 5-th epoch, in
  which case ``resource_attr = "epoch"``.
* The training script needs to know how many resources to spend overall. For
  example, a neural network training script needs to know how many epochs
  to maximally train for. It is best practice to pass this maximum resource value
  as parameter into the script, which is done by making it part of the
  configuration space. In this case, ``max_resource_attr`` is the name of
  the attribute in the configuration space which contains the maximum
  resource value. For example, if your script should train for a maximum of
  100 epochs (the scheduler may stop or pause it before, though), you could
  use ``config_space = dict(..., epochs=100)``, in which case
  ``max_resource_attr = "epochs"``.
* Finally, you can also use ``max_t`` instead of ``max_resource_attr``,
  even though this is not recommended. If you don't want to include the
  maximum resource value in your configuration space, you can pass the
  value directly as ``max_t``. However, this can lead to avoidable errors,
  and may be
  `inefficient for some schedulers <tutorials/multifidelity/mf_setup.html#the-launcher-script>`__.

.. note::
   When creating a multi-fidelity scheduler, we recommend to use
   ``max_resource_attr`` in favour of ``max_t`` or ``max_resource_value``, as
   the latter is error-prone and may be less efficient for some schedulers.

How can I visualize the progress of my tuning experiment with Tensorboard?
==========================================================================

To visualize the progress of Syne Tune in
`Tensorboard <https://www.tensorflow.org/tensorboard>`__, you can pass
the :class:`~syne_tune.callbacks.TensorboardCallback` to the
:class:`~syne_tune.Tuner` object:

.. code:: python

   from syne_tune.callbacks import TensorboardCallback

   tuner = Tuner(
       ...
       callbacks=[TensorboardCallback()],
   )

Note that, you need to install
`TensorboardX <https://github.com/lanpa/tensorboardX>`__ to use this callback:

.. code:: bash

   pip install tensorboardX

This will log all metrics that are reported in your training script via
the ``report(...)`` function. Now, to open Tensorboard, run:

.. code:: bash

   tensorboard --logdir ~/syne-tune/{tuner-name}/tensorboard_output

If you want to plot the cumulative optimum of the metric you want to
optimize, you can pass the ``target_metric`` argument to
class:`syne_tune.callbacks.TensorboardCallback`. This will also report the best
found hyperparameter configuration over time. A complete example is
`examples/launch_tensorboard_example.py <examples.html#visualize-tuning-progress-with-tensorboard>`__.

How can I add a new scheduler?
==============================

This is explained in detail in
`this tutorial <tutorials/developer/README.html>`__, and also in
`examples/launch_height_standalone_scheduler <examples.html#launch-hpo-experiment-with-home-made-scheduler>`__.

Please do consider
`contributing back <https://github.com/awslabs/syne-tune/blob/main/CONTRIBUTING.md>`__
your efforts to the Syne Tune community, thanks!

How can I add a new tabular or surrogate benchmark?
===================================================

To add a new dataset of tabular evaluations, you need to:

* write a blackbox recipe able to regenerate it by extending
  :class:`~syne_tune.blackbox_repository.conversion_scripts.blackbox_recipe.BlackboxRecipe`.
  You need in particular to provide the name of the blackbox, the reference so
  that users are prompted to cite the appropriated paper, and a code that can
  generate it from scratch. See
  :mod:`syne_tune.blackbox_repository.conversion_scripts.scripts.lcbench.lcbench`
  for an example.
* add your new recipe class in
  :mod:`syne_tune.blackbox_repository.conversion_scripts.recipes` to make
  it available in Syne Tune.

Further details are given
`here <tutorials/benchmarking/bm_contributing.html#contributing-a-tabulated-benchmark>`__.

How can I reduce delays in starting trials with the SageMaker backend?
======================================================================

The SageMaker backend executes each trial as a SageMaker training job, which
encurs start-up delays up to several minutes. These delays can be reduced to
about 20 seconds with
`SageMaker managed warm pools <https://docs.aws.amazon.com/sagemaker/latest/dg/train-warm-pools.html>`__,
as is detailed in
`this tutorial <tutorials/benchmarking/bm_sagemaker.html#using-sagemaker-managed-warm-pools>`__
or `this example <examples.html#sagemaker-backend-and-checkpointing>`__. We
strongly recommend to use managed warm pools with the SageMaker backend.

How can I pass lists or dictionaries to the training script?
============================================================

By default, the hyperparameter configuration is passed to the training script
as command line arguments. This precludes parameters from having complex types,
such as lists or dictionaries. The configuration can also be passed as JSON
file, in which case its entries can have any type which is JSON-serializable.
This mode is activated with ``pass_args_as_json=True`` when creating the trial
backend:

.. literalinclude:: ../../examples/launch_height_config_json.py
   :caption: examples/launch_height_config_json.py
   :start-after: if not use_sagemaker_backend:
   :end-before: else:

The trial backend stores the configuration as JSON file and passes its filename
as command line argument. In the training script, the configuration is loaded
as follows:

.. literalinclude:: ../../examples/training_scripts/height_example/train_height_config_json.py
   :caption: examples/training_scripts/height_example/train_height_config_json.py
   :start-at: parser = ArgumentParser()
   :end-at: config = load_config_json(vars(args))

The complete example is
`here <examples.html#pass-configuration-as-json-file-to-training-script>`__.
Note that entries automatically appended to the configuration by Syne Tune, such
as :const:`~syne_tune.constants.ST_CHECKPOINT_DIR`, are passed as command line
arguments in any case.

How can I write extra results for an experiment?
================================================

By default, Syne Tune is writing
`these result files at the end of an experiment <#what-does-the-output-of-the-tuning-contain>`__.
Here, ``results.csv.zip`` contains all data reported by training jobs, along
with time stamps. The contents of this dataframe can be customized, by adding
extra columns to it, as demonstrated in
`examples/launch_height_extra_results.py <examples.html#customize-results-written-during-an-experiment>`__.
