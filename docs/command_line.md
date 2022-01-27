# Syne Tune: Using the Command Line Launcher


## Running HPO on a Benchmark Problem

Instead of launching your experiments using a dedicated Python script, you can
also use the command line launcher. This is particularly convenient when you
want to rapidly compare different options for scheduler or back-end. As we will
see below, you can launch many experiments in sequence or in parallel using the
CL launcher. However, in this section, we will restrict ourselves to launching
a single experiment.

In order to use the CL launcher, your tuning problem has to be defined as a
benchmark and linked into
[benchmark_factory.py](../benchmarking/cli/benchmark_factory.py). Here is
a [tutorial](benchmarks.md) how to do this. In the sequel, we will use one of
the pre-specified benchmarks. First, make sure you have installed the
`gpsearchers` and `benchmarks` dependencies:

```bash
pip install -e .[gpsearchers,benchmarks]
```

Note that the `benchmarks` dependencies are needed only for evaluating the
benchmarks on the same instance you are launching things from. If all your
evaluations are supposed to run remotely, you do not need to install them.

Here is an example for running the CL launcher:

```bash
python benchmarking/cli/launch_hpo.py --scheduler hyperband_stopping --searcher bayesopt \
    --benchmark_name mlp_fashionmnist --scheduler_timeout 120 --local_tuner
```

This is tuning the `mlp_fashionmnist` benchmark (two-layer MLP on FashionMNIST),
using the MOBSTER algorithm, which combines successive halving scheduling with
Bayesian optimization for the search. It uses the local back-end, which runs
on the single instance you are on, and the algorithm stops suggesting new trials
after 120 seconds (and terminates running ones).

The most important arguments for running HPO locally are:
* `benchmark_name`: Name of benchmark to run, as defined in
  [benchmark_factory.py](../benchmarking/cli/benchmark_factory.py)
* `scheduler`: Selects the optimization algorithm. The simplest one is `fifo`.
  For neural network models, multi-fidelity HPO is often a better choice:
  `hyperband_stopping`, `hyperband_promotion` are two variants of asynchronous
  Hyperband / successive halving (`hyperband_stopping` being the simpler option).
  You can also choose Ray Tune schedulers: `raytune_fifo`, `raytune_hyperband`.
* `searcher`: Internally, a scheduler can be configured by a searcher, which
  suggests next candidates to evaluate, and which may fit a surrogate model to
  data observed so far. Use `random` for random choices, `bayesopt` for Gaussian
  process (GP) modelling. If the latter is combined with `hyperband_stopping` or
  `hyperband_promotion`, the data is modelled by multi-task GPs. For the Ray Tune
  schedulers, `scikit-optimize` is used.
* `scheduler_timeout`, `num_trials`: Any of these (or both) define a stopping
  criterion for the optimization. The experiment ends (no new trials are
  started, and running ones are stopped) once the time limit `scheduler_timeout`
  is reached (in secs), or once `num_trials` trials have been completed.
* `max_failures`: The tuning loop is terminated once this many training
  evaluation jobs failed. The default is 1, which is very conservative. Raise
  this value to allow for a certain number of training job failures.
* `local_tuner`: The HPO experiment is run locally on the calling instance.
  This is explained in more detail below.
* `num_workers`: Maximum number of training evaluations executed in parallel.
  For the local back-end, parallel evaluations use separate sub-processes, and
  the number of actual parallel workers is limited by the instance
  configuration (e.g., number of GPUs or vCPUs).
* `experiment_name`: Used as prefix for training job names, and for storing
  results.

A complete list of CL arguments is obtained with:

```bash
python benchmarking/cli/launch_hpo.py --help
```

Some arguments configure the back-end, others the scheduler or its searcher.
See the [scheduler tutorial](schedulers.md) for more details on the latter.

### Controlling the log output

A number of arguments allow you to control the logging output:
* `no_tuner_logging`: By default, the full tuning status is output every 30
  seconds. Setting this flag suppresses that output.
* `debug_log_level`: The default log level is INFO. Setting this flag lowers
  the level to DEBUG, so that more messages are printed. In particular, a
  message is then printed for every report of each trial, which can be helpful
  for FIFO schedulers.
* `no_debug_log`: By default, our built-in schedulers output messages
  concerning the progress of trials. Setting this flag suppresses that output.


## Optimization methods

Our command line launcher currently supports the following optimization methods:
* Random search [`scheduler=fifo`, `searcher=random`]
* Bayesian optimization with Gaussian processes [`scheduler=fifo`, `searcher=bayesopt`]
* Asynchronous Hyperband (ASHA) [`scheduler=hyperband_stopping / hyperband_promotion`, `searcher=random`].
  As proposed in [A System for Massively Parallel Hyperparameter Tuning](https://arxiv.org/abs/1810.05934).
  Strictly speaking, this paper proposes the `scheduler=hyperband_promotion` variant.
  ```bibtex
  @article{
     title={A System for Massively Parallel Hyperparameter Tuning},
     author={Liam Li and Kevin Jamieson and Afshin Rostamizadeh and Ekaterina Gonina and Moritz Hardt and Benjamin Recht and Ameet Talwalkar},
     journal={arXiv preprint arXiv:1810.05934}
  }
  ```
* Model-based Asynchronous Hyperband (MOBSTER) [`scheduler=hyperband_stopping / hyperband_promotion`, `searcher=bayesopt`].
   As proposed in [Model-based Asynchronous Hyperparameter and Neural Architecture Search](https://arxiv.org/abs/2003.10865)
  ```bibtex
  @article{
     title={Model-based Asynchronous Hyperparameter and Neural Architecture Search},
     author={Aaron Klein and Louis C. Tiao and Thibaut Lienart and Cedric Archambeau and Matthias Seeger},
     journal={arXiv preprint arXiv:2003.10865}
  }
  ```
* Ray Tune random search [`scheduler=raytune_fifo`, `searcher=random`]
* Ray Tune Bayesian optimization [`scheduler=raytune_fifo`, `searcher=bayesopt`]. Uses
  [scikit-optimize](https://scikit-optimize.github.io/stable/)
* Ray Tune Asynchronous Hyperband [`scheduler=raytune_hyperband`, `searcher=random`].
  Implements the `hyperband_stopping` variant only.
* Ray Tune Asynchronous Hyperband with scikit-optimize [`scheduler=raytune_hyperband`, `searcher=bayesopt`].
  Uses [scikit-optimize](https://scikit-optimize.github.io/stable/) instead of random
  config choices.


## Launching Many Experiments

Recall the launch command from above:

```bash
python benchmarking/cli/launch_hpo.py --scheduler hyperband_stopping --searcher bayesopt \
    --benchmark_name mlp_fashionmnist --scheduler_timeout 120 --local_tuner
```

This call blocks the instance you are on for the time of the experiment. In
fact, both the tuning loop of the experiment and all training evaluations are
executed on this instance.

What if you want to compare different options, say `bayesopt` with `random`
searcher, or `hyperband_stopping` with `hyperband_promotion` scheduler?
Moreover, outcomes of HPO are quite variable, due to non-convexity and
stochastic search behavior (exploration), and it is best practice to repeat a
setup several times, averaging over results. It is quite painful and slow to do
any of this if each experiment is blocking its instance.

Syne Tune allows you to launch experiments on a remote instance (this requires
[some setup](../README.md#running-on-sagemaker)). For example:

```bash
python benchmarking/cli/launch_hpo.py --scheduler hyperband_stopping --searcher bayesopt \
    --benchmark_name mlp_fashionmnist --scheduler_timeout 120
```

This is the same as the command above, except `--local_tuner` has been dropped.
In this case, the calling instance is not blocked. Instead, tuning loop and all
training evaluations are run as a SageMaker training job. Since this is so
convenient to use, it is the default, and "local tuning" needs to be asked
for explicitly by appending `--local_tuner`.

You can also launch multiple experiments with a single command, which not only
saves typing but also allows you to go for a coffee while your experiments are
getting launched. For example:

```bash
python benchmarking/cli/launch_hpo.py --scheduler hyperband_stopping --searcher bayesopt \
    --benchmark_name mlp_fashionmnist --scheduler_timeout 120 --run_id 0 1 2 3 4
```

This command launches five repetitions of the same experiment, each as its own
SageMaker training job. These experiments have the same parameters, except
`run_id` varies from 0 to 4, and different random seeds are used. Averaging
results over a number of run_id's allows us to robustly compare different
choices. In fact, the same is obtained by running:

```bash
python benchmarking/cli/launch_hpo.py --scheduler hyperband_stopping --searcher bayesopt \
    --benchmark_name mlp_fashionmnist --scheduler_timeout 120 --num_runs 5
```

Beyond repeating the same experiment several times, you can also launch
different variations with a single command. This is useful to compare different
choices, or even to realize a form of brute-force search. To do so, simply
pass list values for certain arguments (most arguments support list values).
If you do this, the combinatorial product of all combinations is iterated over.
For example:

```bash
python benchmarking/cli/launch_hpo.py --scheduler fifo hyperband_stopping \
    --searcher bayesopt random \
    --benchmark_name mlp_fashionmnist \
    --scheduler_timeout 120 --local_tuner
```

This is launching 4 experiments, with `(scheduler, searcher)` taking on values
`(fifo, bayesopt)`, `(hyperband_stopping, bayesopt)`, `(fifo, random)`,
`(hyperband_stopping, random)`.

Iterating over all combinations is not always what you want. Some combinations
may not make sense, or some arguments may depend on each other, so that their
values cannot be varied independently. You can use `argument_groups` in order
to group list value arguments together. The list values of arguments in the
same group are iterated over jointly (think `zip` instead of `product`). For
example:

```bash
python benchmarking/cli/launch_hpo.py \
    --scheduler hyperband_stopping fifo \
    --searcher bayesopt random \
    --max_resource_level 9 27 81 \
    --benchmark_name mlp_fashionmnist \
    --scheduler_timeout 120 --local_tuner \
    --argument_groups "scheduler searcher"
```

This is launching 2 * 3 = 6 experiments, with `(scheduler, searcher,
max_resource_level)` taking on values
`(hyperband_stopping, bayesopt, 9)`, `(fifo, random, 9)`,
`(hyperband_stopping, bayesopt, 27)`, `(fifo, random, 27)`,
`(hyperband_stopping, bayesopt, 81)`, `(fifo, random, 81)`.
Note how `max_resource_level` is in its own (singleton) group, these do not
have to be listed in `argument_groups`.
    
Several groups can be formed. For example:

```bash
python benchmarking/cli/launch_hpo.py \
    --scheduler hyperband_stopping fifo \
    --searcher bayesopt random \
    --benchmark_name mlp_fashionmnist \
    --max_resource_level 27 81 \
    --scheduler_timeout 120 240 --local_tuner \
    --argument_groups "scheduler searcher|max_resource_level scheduler_timeout"
```
    
This is launching 2 * 2 = 4 experiments, with `(scheduler, searcher,
max_resource_level, scheduler_timeout)` taking on values
`(hyperband_stopping, bayesopt, 27, 120)`, `(fifo, random, 27, 120)`,
`(hyperband_stopping, bayesopt, 81, 240)`, `(fifo, random, 81, 240)`.

When multiple experiments are launched in this way, you can use the
`--skip_initial_experiments` argument in order to skip this number of initial
experiments before launching the remaining ones. This is useful if a
previous call failed to launch all intended experiments (e.g., because an
AWS instance limit was reached). If the initial K experiments were in
fact launched, a subsequent call with `--skip_initial_experiments K` will
launch only the remaining ones.

Note that `--benchmark_name` does not support list values. This is because
we also support benchmark-specific command line arguments.

A special argument with list values is `--run_id`, its values must be
distinct nonnegative integers. Here, `--num_runs 5` is short for
`--run_id 0 1 2 3 4`. If `--run_id` is given, `--num_runs` is ignored.
If neither of the two is given, the default is `run_id = 0`.

**Note**: Launching experiments remotely, as SageMaker training jobs, comes
with different properties than running them locally. You can now select the
instance type via `--instance_type`, valid values are
[SageMaker instance types](https://aws.amazon.com/sagemaker/pricing/), such as
'ml.m4.xlarge' (note that leading 'ml.' prefix). The built-in benchmarks come
with suitable defaults for instance types. Also, you do not need the benchmark
dependencies to be installed on your instance (in fact, you can launch
experiments remotely from your laptop).


## How Results are Stored and Downloaded

This is detailed in [Output of a tuning job](../README.md#launching-a-tuning-job).
If tuning is run remotely on SageMaker, results are uploaded to S3 in regular
intervals. When the CLI is used, additional points apply:
* The meta-data in `metadata.json` contains all parameters specified on the
  command line, as well as benchmark-specific defaults. Parameters which are
  neither specified on the CL nor have a benchmark-specific default, are not
  contained. Results can later on be filtered and aggregated based on
  meta-data values.
* When results are uploaded to S3, a slightly different path name convention is
  used. Recall that by default, results are stored in `f"syne-tune/{tuner_name}"`,
  where `tuner_name` has the form `f"{experiment_name}-{datetime}-{hash}"`,
  where `datetime` is the datetime of launch, `hash` is a 3-digit hash. For
  example: `height-tuning-2021-07-02-10-04-37-233`. When using the command line
  launcher, the S3 path is `f"syne-tune/{experiment_name}/{tuner_name}"`
  instead, so that all results for the same `experiment_name` are grouped in a
  subdirectory. This convention can be switched off by `--no_experiment_subdirectory`.
* The S3 bucket for storing results (and checkpoints) can be configured with
  `--s3_bucket`. The default is the bucket assigned to the SM session.

Say you ran a number of experiments with a particular `experiment_name`, say
`myexperiment-1`. You can download relevant results needed for analysis as follows:
```bash
aws s3 sync s3://${BUCKET_NAME}/syne-tune/myexperiment-1/ ~/syne-tune/ \
    --exclude "*" --include "*metadata.json" --include "*results.csv.zip"
```
Note that Syne Tune stores a large number of additional files to S3,
including checkpoints and logs for every trial. A normal `aws s3 sync` takes a
very long time.


## Random Seeds

If you intend to compare different methods with each other, or you like to
ensure that a new version of your code behaves like the previous version, it
is important to control random seeds.

In Syne Tune, a random seed offset is drawn for each call of
`launch_hpo.py`. This seed is printed in the log, and also stored in the experiment
meta-data. It is an integer in `0, ..., 2 ** 32 - 1`. Now, the random seed used
for each experiment is this random seed offset plus `run_id` modulo `2 ** 32`, so
that two experiments started with the same call have the same seed iff their
`run_id` is the same. For example, if you compare random search with Bayesian
optimization, the initial random choices are the same for the same `run_id`,
allowing for a paired comparison, but inherent randomness is quantified by using
a number of `run_id`'s.

The random seed offset can be specified with `--random_seed_offset`. This way,
you can compare different variants without launching them with a single call,
or even do comparisons across different `experiment_name`'s.

Note that the seeds controlled this way only affect random choices in Syne Tune,
not in the training code to be tuned (e.g., weight initialization, ordering of
data batches, data set splits).


## Different Back-Ends

All examples above make use of the `local` back-end in order to execute
training evaluations. For the local back-end, the tuning loop and all parallel
training evaluations ("workers") are running on the same instance. For local
tuning, this is the instance where the CL launcher is started. For remote
tuning, it is a SageMaker training job on a different instance.

Syne Tune also supports the `sagemaker` back-end. For example:

```bash
python benchmarking/cli/launch_hpo.py --scheduler hyperband_stopping --searcher bayesopt \
    --benchmark_name mlp_fashionmnist --scheduler_timeout 120 --local_tuner \
    --backend sagemaker
```

This launches the now familiar experiment using the `sagemaker` back-end, which
means that *every single* training evaluation is executed as a SageMaker
training job. Due to `local_tuner`, the tuning loop is run on the calling
instance, which blocks until the experiment is finished.

The `sagemaker` back-end is most suitable for expensive benchmarks. Each
evaluation is run on a separate instance and can make use of all computational
resources there (GPU and CPU cores, memory). It is also the most robust
back-end supported as of now: tuning loop and each training evaluation run
on different instances and interact via standard SageMaker channels. Finally,
the `sagemaker` back-end is most flexible when it comes to training evaluation
code. It can handle essentially everything which can be started as SageMaker
training job. A drawback of this back-end is that considerable over-heads for
starting and stopping SageMaker training jobs are suffered for every training
evaluation. If your model trains rapidly, the `local` back-end is a better
choice.

Once the `sagemaker` back-end is chosen, a number of additional arguments become
relevant:
* `instance_type`: Instance type for training evaluations. Note that each
  evaluation can make use of all resources there. Suitable defaults for
  built-in benchmarks.
* `sagemaker_execution_role`: If you launch experiments on a SageMaker notebook
  instance, the role is determined automatically. Otherwise, you have to
  specify it here.
* `image_uri`: Your training evaluation code will have dependencies, which may
  not be easy to setup (e.g., if GPU computation is needed). SageMaker supports
  a number of *frameworks* (e.g., `PyTorch`, `HuggingFace`, `TensorFlow`). If
  these dependencies cover most of what your code needs, you are set (see
  [benchmarks tutorial](benchmarks.md) for details). If you have special needs
  or do not want to use a SageMaker framework, you will have to build a Docker
  image. Once
  this is done and uploaded to ECR, its URI is passed using this argument.
* `s3_bucket`: S3 bucket where checkpoints are stored. If not given, the
  default bucket for the session is used.


### SageMaker frameworks

A very convenient aspect of SageMaker are its
[frameworks](https://sagemaker.readthedocs.io/en/stable/frameworks/index.html).
Essentially, a framework manages dependencies for customers, which may include
AWS specific optimizations or simplifications. Not making use of a framework
often means that you have to create your own Docker image, or at least manage
your dependencies on top of a more generic framework.

Syne Tune currently supports frameworks only with the `sagemaker` back-end.
For remote tuning with the local back-end, while each tuning experiment maps to
a SageMaker training job, these jobs use the PyTorch framework together with a
pre-built image containing the Syne Tune dependencies. If your benchmark
code requires additional dependencies on top of PyTorch, you can specify them
in `dependencies.txt`. For example, if your training script uses Hugging Face,
you need to add `transformers` and `datasets` to `dependencies.txt`. If your benchmark uses
TensorFlow or requires other specific dependencies which cannot be based on top
of PyTorch, you currently cannot use remote tuning with the local back-end.
More details are given in the [benchmarks tutorial](benchmarks.md).


### Remote Tuning with SageMaker Back-end

You can also combine remote tuning with the `sagemaker` back-end, for example:

```bash
python benchmarking/cli/launch_hpo.py --scheduler hyperband_stopping --searcher bayesopt \
    --benchmark_name mlp_fashionmnist --scheduler_timeout 120 \
    --backend sagemaker
```

This call is non-blocking. Both the tuning loop as well as every single training
evaluation are executed as separate SageMaker training jobs. In this case,
`instance_type` determines the instance type for training evaluations (with good
defaults for each built-in benchmark), but what about the tuning loop? Its
instance type can be specified by `tuner_instance_type` (whose default is a
reasonably cheap CPU instance). For example:

```bash
python benchmarking/cli/launch_hpo.py --scheduler hyperband_stopping --searcher bayesopt \
    --benchmark_name mlp_fashionmnist --scheduler_timeout 120 \
    --backend sagemaker --tuner_instance_type ml.c5.xlarge
```

If your HPO decision making algorithm needs some beefy computations, you might
want to select a more powerful instance type.
