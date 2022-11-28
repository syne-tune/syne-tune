# Syne Tune FAQ

* [Why should I use Syne Tune?](#why-should-i-use-syne-tune)
* [What are the different installations options supported?](#what-are-the-different-installations-options-supported)
* [How can I run on AWS and SageMaker?](#how-can-i-run-on-aws-and-sagemaker)
* [What are the metrics reported by default when calling the `Reporter`?](#what-are-the-metrics-reported-by-default-when-calling-the-reporter)
* [How can I utilize multiple GPUs?](#how-can-i-utilize-multiple-gpus)
* [What is the default mode when performing optimization?](#what-is-the-default-mode-when-performing-optimization)
* [How are trials evaluated on a local machine?](#how-are-trials-evaluated-on-a-local-machine)
* [What does the output of the tuning contain?](#what-does-the-output-of-the-tuning-contain)
* [Where can I find the output of the tuning?](#where-can-i-find-the-output-of-the-tuning)
* [How can I enable trial checkpointing?](#how-can-i-enable-trial-checkpointing)
* [Which schedulers make use of checkpointing?](#which-schedulers-make-use-of-checkpointing)
* [Is the tuner checkpointed?](#is-the-tuner-checkpointed)
* [Where can I find the output of my trials?](#where-can-i-find-the-output-of-my-trials)
* [How can I plot the results of a tuning?](#how-can-i-plot-the-results-of-a-tuning)
* [How can I specify additional tuning metadata?](#how-can-i-specify-additional-tuning-metadata)
* [How do I append additional information to the results which are stored?](#how-do-i-append-additional-information-to-the-results-which-are-stored) 
* [I don’t want to wait, how can I launch the tuning on a remote machine?](#i-dont-want-to-wait-how-can-i-launch-the-tuning-on-a-remote-machine)
* [How can I run many experiments in parallel?](#how-can-i-run-many-experiments-in-parallel)
* [How can I access results after tuning remotely?](#how-can-i-access-results-after-tuning-remotely)
* [How can I specify dependencies to remote launcher or when using the SageMaker backend?](#how-can-i-specify-dependencies-to-remote-launcher-or-when-using-the-sagemaker-backend)
* [How can I benchmark different methods?](#how-can-i-benchmark-different-methods)
* [What different schedulers do you support? What are the main differences between them?](#what-different-schedulers-do-you-support-what-are-the-main-differences-between-them)
* [How do I define the search space?](#how-do-i-define-the-search-space) 
* [How can I visualize the progress of my tuning experiment with Tensorboard?](#how-can-i-visualize-the-progress-of-my-tuning-experiment-with-tensorboard)
* [How can I add a new scheduler?](#how-can-i-add-a-new-scheduler)
* [How can I add a new tabular or surrogate benchmark?](#how-can-i-add-a-new-tabular-or-surrogate-benchmark)


## Why should I use Syne Tune?

HPO is an important problem since many years, with a healthy number of commercial
and open source tools available. Notable examples for open source tools are
[Ray Tyne](https://docs.ray.io/en/latest/tune/index.html) and
[Optuna](https://optuna.readthedocs.io/en/stable/). Here are some reasons why you
may prefer Syne Tune over these alternatives:
* Lightweight and platform-agnostic: Syne Tune is designed to work with different
  execution back-ends, so you are not locked into a particular distributed system
  architecture. Syne Tune runs with minimal dependencies.
* Wide range of modalities: Syne Tune supports multi-fidelity HPO, constrained HPO,
  multi-objective HPO, transfer tuning, cost-aware HPO, population based training.
* Simple, modular design: Rather than wrapping all sorts of other HPO frameworks,
  Syne Tune provides simple APIs and scheduler templates, which can easily be
  [extended to your specific needs](tutorials/developer/README.md).
* Industry-strength Bayesian optimization: Syne Tune has special support for
  [Gaussian process based Bayesian optimization](tutorials/basics/basics_bayesopt.md).
  The same code powers modalities like multi-fidelity HPO, constrained HPO, or
  cost-aware HPO, having been tried and tested for several years in SageMaker
  services.
* Special support for researchers: Syne Tune allows for rapid development and
  comparison between different tuning algorithms. Its
  [blackbox repository and simulator back-end](tutorials/multifidelity/mf_setup.md)
  run realistic simulations of experiments many times faster than real time.
  [Benchmarking](tutorials/benchmarking/README.md) is simple and efficient.

If you are an AWS customer, there are additional good reasons to use Syne Tune over
the alternatives:
* If you use AWS services or SageMaker frameworks day to day, Syne Tune works
  out of the box and fits into your normal workflow.
* Syne Tune is developed in collaboration with the team behind the
  [Automatic Model Tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html)
  service.


## What are the different installations options supported?

To install Syne Tune with minimal dependencies from pip, you can simply do:

```bash
pip install 'syne-tune[core]'
```

If you want in addition to install our own Gaussian process based optimizers, Ray Tune or Bore optimizer, 
you can run `pip install 'syne-tune[X]'` where `X` can be 
* `gpsearchers`: For built-in Gaussian process based optimizers
* `aws`: AWS SageMaker dependencies
* `raytune`: For Ray Tune optimizers, installs all Ray Tune dependencies
* `benchmarks`: For installing dependencies required to run all benchmarks
* `blackbox-repository`: Blackbox repository for simulated tuning
* `yahpo`: YAHPO Gym surrogate blackboxes
* `kde`: For BOHB
* `botorch`: Bayesian optimization from BOTorch
* `dev`: For developers who want to extend Syne Tune
* `extra`: For installing all the above
* `bore`: For Bore optimizer

For instance, `pip install 'syne-tune[gpsearchers]'` will install Syne Tune
along with many built-in Gaussian process optimizers.

To install the latest version from git, run the following:

```bash
pip install git+https://github.com/awslabs/syne-tune.git
```

For local development, we recommend using the following setup which will enable
you to easily test your changes:

```bash
git clone https://github.com/awslabs/syne-tune.git
cd syne-tune
python3 -m venv st_venv
. st_venv/bin/activate
pip install --upgrade pip
pip install -e '.[extra]'
```

This installs everything in a virtual environment `st_venv`. Remember to activate
this environment before working with Syne Tune. We also recommend building the
virtual environment from scratch now and then, in particular when you pull a new
release, as dependencies may have changed.


## How can I run on AWS and SageMaker?

If you want to launch experiments on SageMaker rather than on your local machine, you will need access to AWS and SageMaker on your machine.
Make sure that:

* `awscli` is installed (see [this link](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html))
* AWS credentials have been set properly (see [this link](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html)).
* The necessary SageMaker role has been created (see [this page](https://docs.aws.amazon.com/glue/latest/dg/create-an-iam-role-sagemaker-notebook.html) for instructions. If you've created a SageMaker notebook in the past, this role should already have been created for you).

The following command should run without error if your credentials are available:

```bash
python -c "import boto3; print(boto3.client('sagemaker').list_training_jobs(MaxResults=1))"
```

You can also run the following example that evaluates trials on SageMaker to test your setup.

```bash
python examples/launch_height_sagemaker.py
```


## What are the metrics reported by default when calling the `Reporter`?

Whenever you call the reporter to log a result, the worker time-stamp, the worker time since the creation of the reporter and the number of times the reporter was called are logged under the fields `st_worker_timestamp`,  `st_worker_time`, and `st_worker_iter`. In addition, when running on SageMaker, a dollar-cost estimate is logged under the field `st_worker_cost`.

To see this behavior, you can simply call the reporter to see those metrics:

```python
from syne_tune.report import Reporter
reporter = Reporter()
for step in range(3):
   reporter(step=step, metric=float(step) / 3)
   
# [tune-metric]: {"step": 0, "metric": 0.0, "st_worker_timestamp": 1644311849.6071281, "st_worker_time": 0.0001048670000045604, "st_worker_iter": 0}
# [tune-metric]: {"step": 1, "metric": 0.3333333333333333, "st_worker_timestamp": 1644311849.6071832, "st_worker_time": 0.00015910100000837701, "st_worker_iter": 1}
# [tune-metric]: {"step": 2, "metric": 0.6666666666666666, "st_worker_timestamp": 1644311849.60733, "st_worker_time": 0.00030723599996917983, "st_worker_iter": 2}
```


## How can I utilize multiple GPUs?

To utilize multiple GPUs you can either use the backend `LocalBackend` which
will run on the GPU available in a local machine. You can also run on a remote
AWS machine with multiple GPUs using the local backend and the remote launcher,
see [here](#i-dont-want-to-wait-how-can-i-launch-the-tuning-on-a-remote-machine),
or run with the SageMaker backend which spins-up one training job per trial.

When evaluating trials on a local machine with `LocalBackend`, by default each trial is allocated to the least occupied GPU by setting `CUDA_VISIBLE_DEVICES` environment variable. 


## What is the default mode when performing optimization?

The default mode is `"min"` when performing optimization, so the target metric is minimized. The mode can be configured when instantiating a scheduler.


## How are trials evaluated on a local machine?

When trials are executed locally (e.g. when `LocalBackend` is used), each trial is evaluated as a different sub-process.
As such the number of concurrent configurations evaluated at the same time (set by `n_workers`) should
account for the capacity of the machine where the trials are executed.


## Where can I find the output of the tuning?

When running locally, the output of the tuning is saved under `~/syne-tune/{tuner-name}/` by default. 
When running remotely on SageMaker, the output of the tuning is saved under `/opt/ml/checkpoints/` by default and
the tuning output is synced regularly to  `s3://{sagemaker-default-bucket}/syne-tune/{tuner-name}/`.


## How can I change the default output folder where tuning results are stored?

To change the path where tuning results are written, you can set the environment variable `SYNETUNE_FOLDER` to the 
folder that you want.

For instance, the following runs a tuning where results tuning files are written under `~/new-syne-tune-folder`: 

```bash
export SYNETUNE_FOLDER="~/new-syne-tune-folder"
python examples/launch_height_baselines.py
```

You can also do the following for instance to permanently change the output folder of Syne Tune:

```bash
echo 'export SYNETUNE_FOLDER="~/new-syne-tune-folder"' >> ~/.bashrc && source ~/.bashrc
```


## What does the output of the tuning contain?

Syne Tune stores the following files `metadata.json`, `results.csv.zip`, and `tuner.dill`
which are respectively metadata of the tuning job, results obtained at each
time-step and state of the tuner.


## How can I enable trial checkpointing?

Since trials may be paused and resumed (either by schedulers or when using
spot-instances), the user may checkpoint intermediate results to avoid starting
computation from scratch. Model outputs and checkpoints must be written into a
specific local path given by the command line argument `st_checkpoint_dir`.
Saving/loading model checkpoint from this directory enables to save/load the state
when the job is stopped/resumed (setting the folder correctly and uniquely per
trial is the responsibility of the backend), see
[checkpoint_example.py](../examples/training_scripts/checkpoint_example/checkpoint_example.py)
for a working example of a tuning script with checkpointing enabled.

When using SageMaker backend or tuning remotely, we use the
[SageMaker checkpoint mechanism](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html)
under the hood to sync local checkpoints to s3. Checkpoints are synced to
`s3://{sagemaker-default-bucket}/syne-tune/{tuner-name}/{trial-id}/`, where
`sagemaker-default-bucket` is the default bucket for SageMaker.

There are some convenience functions which help you to implement checkpointing
for your training script. Have a look at the example
[lstm_wikitext2.py](../benchmarking/training_scripts/lstm_wikitext2/lstm_wikitext2.py):
* Checkpoints have to be written at the end of certain epochs (namely those
  after which the scheduler may pause the trial). This is dealt with by
  `checkpoint_model_at_rung_level(config, save_model_fn, epoch)`. Here,
  `epoch` is the current epoch, allowing the function to decide whether to
  checkpoint or not. `save_model_fn` stores the current mutable state
  along with `epoch` to a local path (see below). Finally, `config` contains
  arguments provided by the scheduler (see below).
* Before the training loop starts (and optionally), the mutable state to start
  from has to be loaded from a checkpoint. This is done by
  `resume_from_checkpointed_model(config, load_model_fn)`. If the
  checkpoint has been loaded successfully, the training loop may start with
  epoch `resume_from + 1` instead of `1`. Here, `load_model_fn` loads the
  mutable state from a checkpoint in a local path, returning its `epoch`
  value if successful, which is returned as `resume_from`.

In general, `load_model_fn` and `save_model_fn` have to be provided as part of
the script. For most PyTorch models, you can use `pytorch_load_save_functions`
to this end. In general, you will want to include the model, the optimizer,
and the learning rate scheduler. In our example above, optimizer and
learning rate scheduler are home-made, the state of the latter is contained in
`mutable_state`.

Finally, the scheduler provides additional information about checkpointing in
`config`. You don't have to worry about this:
`add_checkpointing_to_argparse(parser)` adds corresponding arguments to the parser.


## Which schedulers make use of checkpointing?

Checkpointing means storing the state of a trial (i.e., model parameters, optimizer
or learning rate scheduler parameters), so that it can be paused and potentially
resumed at a later point in time, without having to start training from scratch.
The following schedulers make use of checkpointing:

* Promotion-based Hyperband: `HyperbandScheduler(type='promotion', ...)`, as well
  as other asynchronous multi-fidelity schedulers.
  The code runs without checkpointing, but in this case, any trial which is resumed
  is started from scratch. For example, if a trial was paused after 9 epochs of
  training and is resumed later, training starts from scratch and the first 9
  epochs are wasted effort. Moreover, extra variance is introduced by starting from
  scratch, since weights may be initialized differently. It is not recommended
  running promotion-based Hyperband without checkpointing.
* Population-based training: `PopulationBasedTraining`
  PBT does not work without checkpointing.
* Synchronous Hyperband: `SynchronousGeometricHyperbandScheduler`, as well as
  other synchronous multi-fidelity schedulers.
  This code runs without checkpointing, but wastes effort in the same sense as
  promotion-based asynchronous Hyperband


## Is the tuner checkpointed?

Yes. When performing the tuning, the tuner state is regularly saved on the experiment path under `tuner.dill`
(every 10 seconds which can be configured with `results_update_interval`).
This allows to use spot-instances when running a tuning remotely with the remote launcher. It also allows to 
resume a past experiment or analyse the state of scheduler at any point.


## Where can I find the output of my trials?

When running  `LocalBackend` locally, results of trials are saved under `~/syne-tune/{tuner-name}/{trial-id}/` and contains the following files:

* `config.json`: configuration that is being evaluated in the trial
* `std.err`: standard error 
* `std.out`: standard output

In addition all checkpointing files used by a training script such as intermediate model checkpoint will also be located there.
This is exemplified in the following example:

```
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
```

When running tuning remotely with the remote launcher, only `config.json`, `metadata.json`, `results.csv.zip` and `tuner.dill` 
are synced with S3 unless `store_logs_localbackend` in which case the trial logs and informations are also persisted.


## How can I plot the results of a tuning?

The easiest way to plot the result of a tuning experiment is to call the following:

```python
tuner = Tuner(
    ...
    tuner_name="plot-results-demo",
)
tuner.run()
tuning_experiment = load_experiment(tuner.name)
tuning_experiment.plot()
```

This generates a plot of the best value found over time. Note that this you can also plot the results while the experiment is running as results are updated continuously.


## How can I specify additional tuning metadata?

By default, Syne Tune stores the time, the names and modes of the metric being tuner, the name of the entrypoint, the name backend and the scheduler name.
You can also add custom metadata to your tuning job by setting `metadata` in `Tuner` as follow:

```python
tuner = Tuner(
    ...
    tuner_name="plot-results-demo",
    metadata={"tag": "special-tag", "user": "alice"},
)
```

All Syne Tune and user metadata are saved when the tuner starts under `metadata.json`.


## How do I append additional information to the results which are stored? 

Results are processed and stored by callbacks passed to `Tuner`, in particular see 
[tuner_callback.py](../syne_tune/tuner_callback.py#L80). 
In order to add more information to these results, you can inherit from `StoreResultsCallback`. 
A good example is given in 
[searcher_callback.py](../syne_tune/optimizer/schedulers/searchers/searcher_callback.py#L63).

If you run experiments with tabulated benchmarks using the `SimulatorBackend`, as demonstrated in 
[launch_nasbench201_simulated.py](../examples/launch_nasbench201_simulated.py), results are stored by `SimulatorCallback` instead, and you need to inherit from this class, as shown in 
[searcher_callback.py](../syne_tune/optimizer/schedulers/searchers/searcher_callback.py#L88).


## I don’t want to wait, how can I launch the tuning on a remote machine?

Remote launching of experiments has a number of advantages:
* The machine you are working on is not blocked
* You can launch many experiments in parallel
* You can launch experiments with any instance type you like, without having to
  provision them yourselves

You can use the remote launcher to launch an experiment on a remote machine. The
remote launcher supports both `LocalBackend` and `SageMakerBackend`. In the
former case, multiple trials will be evaluated on the remote machine (one use-case
being to use a beefy machine), in the latter case trials will be evaluated as
separate SageMaker training jobs. An example for running the remote launcher is
given in
[launch_height_sagemaker_remotely.py](../examples/launch_height_sagemaker_remotely.py).

Remote launching for benchmarking (i.e., running many remote experiments in order
to compare multiple methods) is detailed in
[this tutorial](tutorials/benchmarking/README.md).


## How can I run many experiments in parallel?

You can remotely launch any number of experiments, which will then run in parallel,
as detailed in
[this tutorial](tutorials/benchmarking/README.md), see also these examples:
* Local backend: [launch_remote.py](../benchmarking/nursery/launch_local/launch_remote.py)
* Simulator backend: [launch_remote.py](../benchmarking/nursery/benchmark_dehb/launch_remote.py)
* SageMaker backend: [launch_remote.py](../benchmarking/nursery/launch_sagemaker/launch_remote.py)

Another example is given in [benchmark_loop](../benchmarking/benchmark_loop).


## How can I access results after tuning remotely?

You can either call `load_experiment("{tuner-name}")` which will download files from s3 if the experiment is not found locally. You can also sync directly files from s3 under `~/syne-tune/` folder in batch for instance by running:

```bash
aws s3 sync s3://{sagemaker-default-bucket}/syne-tune/{tuner-name}/ ~/syne-tune/  --include "*"  --exclude "*tuner.dill"
```

To get all results without the tuner state (you can ommit the include and exclude if you also want to include the tuner state).


## How can I specify dependencies to remote launcher or when using the SageMaker backend?

When you run remote code, you often need to install packages (e.g. `scipy`) or have custom code available.

* To install packages, you can add a file `requirements.txt` in the same folder
  as your endpoint script. All those packages will be installed by SageMaker
  when docker container starts.
* To include custom code (for instance a library that you are working on), you
  can set the parameter `dependencies` on the remote launcher or on a SageMaker
  framework to a list of folders. The folders indicated will be compressed,
  sent to s3 and added to the python path when the container starts. You can
  see [launch_remote.py](../benchmarking/benchmark_loop/launch_remote.py#L28)
  for an example setting dependencies in a SageMaker estimator.


## How can I benchmark different methods?

The most flexible way to do so is to write a custom launcher script, as detailed in
[this tutorial](tutorials/benchmarking/README.md), see also these examples:
* Local backend: [launch_remote.py](../benchmarking/nursery/launch_local/launch_remote.py)
* Simulator backend: [launch_remote.py](../benchmarking/nursery/benchmark_dehb/launch_remote.py)
* SageMaker backend: [launch_remote.py](../benchmarking/nursery/launch_sagemaker/launch_remote.py)
* Fine-tuning transformers: [launch_remote.py](../benchmarking/nursery/fine_tuning_transformer_glue/launch_remote.py)
* Hyper-Tune: [launch_remote.py](../benchmarking/nursery/benchmark_hypertune/launch_remote.py)


## What different schedulers do you support? What are the main differences between them?

We refer to HPO algorithms as *schedulers*. A scheduler decides which configurations
to assign to new trials, but also when to stop a running or resume a paused trial.
Some schedulers delegate the first decision to a *searcher*. The most important
differences between schedulers in the single-objective case are:

* Does the scheduler stop trials early or pause and resume trials (`HyperbandScheduler`)
  or not (`FIFOScheduler`). The former requires a resource dimension (e.g., number
  of epochs; size of training set) and slightly more elaborate reporting (e.g.,
  evaluation after every epoch), but can outperform the latter by a large margin.
* Does the searcher suggest new configurations by uniform random sampling
  (`searcher='random'`) or by sequential model-based decision-making
  (`searcher='bayesopt'`, `searcher='kde'`). The latter can be more expensive if a
  lot of trials are run, but can also be more sample-efficient.

An overview of this landscape is given [here](schedulers.md).

Here is a [tutorial for multi-fidelity schedulers](tutorials/multifidelity/README.md).
Further schedulers provided by Syne Tune include:

* [Population based training (PBT)](../examples/launch_pbt.py)
* [Multi-objective asynchronous successive halving (MOASHA)](../examples/launch_moasha_instance_tuning.py)
* [Constrained Bayesian optimization](../examples/launch_bayesopt_constrained.py)
* [Bayesian optimization by density-ratio estimation (BORE)](../syne_tune/optimizer/baselines.py#L172)
* [Regularized evolution](../syne_tune/optimizer/baselines.py#L185)
* [Median stopping rule](../syne_tune/optimizer/schedulers/median_stopping_rule.py)
* [Synchronous Hyperband](tutorials/multifidelity/mf_syncsh.md)
* [Differential Evolution Hyperband (DEHB)](tutorials/multifidelity/mf_sync_model.md)
* [Hyper-Tune](tutorials/multifidelity/mf_async_model.md)
* [Transfer learning schedulers](../examples/launch_nas201_transfer_learning.py)
* [Wrappers for Ray Tune schedulers](../examples/launch_height_ray.py)

Most of those methods can be accessed with short names by from
[baselines.py](../syne_tune/optimizer/baselines.py).


## How do I define the search space? 

While the training script defines the function to be optimized, some care needs
to be taken to define the search space for the hyperparameter optimization
problem. This being a global optimization problem without gradients easily
available, it is most important to reduce the number of parameters. Some advice
is given [here](search_space.md).

A powerful approach is to run experiments in parallel. Namely, split your
hyperparameters into groups A, B, such that HPO over B is tractable. Draw a set
of N configurations from A at random, then start N HPO experiments in parallel,
where in each of them the search space is over B only, while the parameters in A
are fixed. Syne Tune supports massively parallel experimentation, see
[this tutorial](tutorials/benchmarking/README.md).


## How can I visualize the progress of my tuning experiment with Tensorboard?

To visualize the progress of Syne Tune in
[Tensorboard](https://www.tensorflow.org/tensorboard), you can pass the
`TensorboardCallback` to the `Tuner` object:

```python
from syne_tune.callbacks import TensorboardCallback

tuner = Tuner(
    ...
    callbacks=[TensorboardCallback()],
)
```
Note that, you need to install [TensorboardX](https://github.com/lanpa/tensorboardX)
to use this callback. You can install it by:
```bash
pip install tensorboardX
```
This will log all metrics that are reported in your training script via the `report(...)`
function. Now, to open Tensorboard, run:

```bash
tensorboard --logdir ~/syne-tune/{tuner-name}/tensorboard_output
```

If you want to plot the cumulative optimum of the metric you want to optimize,
you can pass the `target_metric` argument to `TensorboardCallback`. This will
also report the best found hyperparameter configuration over time.


## How can I add a new scheduler?

This is explained in detail in [this tutorial](tutorials/developer/README.md).
Please do consider [contributing back](../CONTRIBUTING.md) your efforts to the
Syne Tune community, thanks!


## How can I add a new tabular or surrogate benchmark?

To add a new dataset of tabular evaluations, you need to
* write a blackbox recipe able to regenerate it by extending
  [`BlackboxRecipe`](../syne_tune/blackbox_repository/conversion_scripts/blackbox_recipe.py). 
  You need in particular to provide the name of the blackbox, the reference
  so that users are prompted to cite the appropriated paper, and a code 
  that can generate it from scratch. See
  [`lcbench.py`](../syne_tune/blackbox_repository/conversion_scripts/scripts/lcbench/lcbench.py) 
  for an example.
* add your new recipe class in 
  [`recipes.py`](../syne_tune/blackbox_repository/conversion_scripts/recipes.py) 
  to make it available in Syne Tune.

Further details are given [here](tutorials/benchmarking/bm_contributing.md#contributing-a-tabulated-benchmark).
