# Syne Tune

[![Release](https://img.shields.io/badge/release-0.2-brightgreen.svg)](https://pypi.org/project/syne-tune/)
[![Python Version](https://img.shields.io/badge/3.7%20%7C%203.8%20%7C%203.9-brightgreen.svg)](https://pypi.org/project/syne-tune/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://pepy.tech/badge/syne-tune/month)](https://pepy.tech/project/syne-tune)

This package provides state-of-the-art distributed hyperparameter optimizers (HPO) where trials 
 can be evaluated with several trial backend options (local backend to evaluate trials locally;
SageMaker to evaluate trials as separate SageMaker training jobs; a simulation backend to quickly benchmark parallel 
asynchronous schedulers).

## Installing

To install Syne Tune from pip, you can simply do:

```bash
pip install 'syne-tune'
```

This will install a bare-bone version. If you want in addition to install our own Gaussian process based optimizers, Ray Tune or Bore optimizer, 
you can run `pip install 'syne-tune[X]'` where `X` can be 
* `gpsearchers`: For built-in Gaussian process based optimizers
* `raytune`: For Ray Tune optimizers
* `benchmarks`: For installing all dependencies required to run all benchmarks
* `extra`: For installing all the above
* `bore`: For Bore optimizer

For instance, `pip install 'syne-tune[gpsearchers]'` will install Syne Tune along with many built-in Gaussian process 
optimizers.

To install the latest version from git, run the following:

```bash
pip install git+https://github.com/awslabs/syne-tune.git
```

For local development, we recommend to use the following setup which will enable you to easily test your changes: 

```bash
pip install --upgrade pip
git clone git@github.com:awslabs/syne-tune.git
cd syne-tune
pip install -e '.[extra]'
```

To run unit tests, simply run `pytest` in the root of this repository.

To run all tests whose name begins with `test_async_scheduler`, you can use the following
```bash
pytest -k test_async_scheduler
```

## How to enable tuning and tuning script conventions

This section describes how to enable tuning an endpoint script. In particular, we describe:

1. how hyperparameters are transmitted from the “tuner” to the user script function
2. how the user communicates metrics to the “tuner” script (which depends on a backend implementation)
3. how does the user enables checkpointing to pause/resume trial tuning jobs?

**Hyperparameters.** Hyperparameters are passed through command line arguments as in SageMaker.
For instance, for a hyperparameters num_epochs:

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, required=True)
args, _ = parser.parse_known_args()
for i in range(1, args.num_epochs + 1):
  ... # do something
```

**Communicating metrics.** 
You should call a function to report metrics after each epochs or at the end of the trial. 
For example:

```python
from syne_tune.report import Reporter
report = Reporter()
for epoch in range(1, num_epochs + 1):
   # ... do something
   train_acc = compute_accuracy()
   report(train_acc=train_acc, epoch=epoch)
```
reports artificial results obtained in a dummy loop. 
In addition to user metrics, Syne Tune will automatically add the following metrics:

* `st_worker_timestamp`: the time stamp when report was called
* `st_worker_time`: the total time spent when report was called since the creation of the reporter
* `st_worker_cost` (only when running on SageMaker): the dollar-cost spent since the creation of the reporter

**Model output and checkpointing (optional).** 
Since trials may be paused and resumed (either by schedulers or when using spot-instances), 
the user has the possibility to checkpoint intermediate results. Model outputs and 
checkpoints must be written into a specific local path given by the command line argument 
`st_checkpoint_dir`. Saving/loading model checkpoint from this directory enables to save/load
 the state when the job is stopped/resumed (setting the folder correctly and uniquely per
  trial is the responsibility of the trial backend), see 
  [checkpoint_example.py](examples/training_scripts/checkpoint_example/checkpoint_example.py) to see a fully
   working example of a tuning script with checkpoint enabled.

Under the hood, we use [SageMaker checkpoint mechanism](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html) 
to enable checkpointing when running tuning remotely or when using the SageMaker
backend. Checkpoints are saved in `s3://{s3_bucket}/syne-tune/{tuner-name}/{trial-id}/`,
where `s3_bucket` can be configured (defaults to `default_bucket` of the
session).

We refer to [checkpoint_example.py](examples/training_scripts/checkpoint_example/checkpoint_example.py) for a complete
 example of a script with checkpoint enabled.

Many other examples of scripts that can be tuned are are available in 
[examples/training_scripts](examples/training_scripts).

## Launching a tuning job

**Tuning options.** At a high-level a tuning consists in a tuning-loop that evaluates different trial in parallel and only let the top ones continue. This loop continues until a stopping criterion is met (for instance a maximum wallclock-time) and each time a worker is available asks a scheduler (an HPO algorithm) to decide which trial should be evaluated next. The execution of the trial is done on a backend. 
The pseudo-code of an HPO loop is as follow:

```python
def hpo_loop(hpo_algorithm, trial_backend):
    while not_done():
        if worker_is_free():
            config = hpo_algorithm.suggest()
            trial_backend.start_trial(config)
        for result in trial_backend.fetch_new_results():
            decision = hpo_algorithm.on_trial_result(result)
            if decision == "stop":
                trial_backend.stop_trial(result.trial)
```
By changing the trial backend, users can decide whether the trial should be evaluated in a local machine, 
whether the trial should be executed on SageMaker with a separate training job or whether the trial should 
be evaluated on a cluster of multiple machines (available as a separate package for now).

Below is a minimal example showing how to tune a script `train_height.py` with Random-search:

```python
from pathlib import Path
from syne_tune.config_space import randint
from syne_tune.backend.local_backend import LocalBackend
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner

config_space = {
    "steps": 100,
    "width": randint(0, 20),
    "height": randint(-100, 100)
}

# path of a training script to be tuned
entry_point = Path(__file__).parent / "training_scripts" / "height_example" / "train_height.py"

# Local back-end
trial_backend = LocalBackend(entry_point=str(entry_point))

# Random search without stopping
scheduler = FIFOScheduler(
    config_space,
    searcher="random",
    mode="min",
    metric="mean_loss",
)

tuner = Tuner(
    trial_backend=trial_backend,
    scheduler=scheduler,
    stop_criterion=StoppingCriterion(max_wallclock_time=30),
    n_workers=4,
)

tuner.run()
```

An important part of this script is the definition of `config_space`, the
configuration space (or search space). [This tutorial](docs/search_space.md)
provides some advice on this choice.

Using the local backend `LocalBackend(entry_point=...)` allows to run the trials (4 at the same time) 
on the local machine. If instead, users prefer to evaluate trials on SageMaker, then SageMaker backend 
can be used which allow to tune any SageMaker Framework (see 
[launch_height_sagemaker.py](examples/launch_height_sagemaker.py) for an example), 
here is one example to run a PyTorch estimator on a GPU

```python
from sagemaker.pytorch import PyTorch
from syne_tune.backend import SageMakerBackend
from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role

trial_backend = SageMakerBackend(
    # we tune a PyTorch Framework from Sagemaker
    sm_estimator=PyTorch(
        entry_point="path_to_your_entrypoint.py",
        instance_type="ml.p2.xlarge",
        instance_count=1,
        role=get_execution_role(),
        max_run=10 * 60,
        framework_version='1.7.1',
        py_version='py3',
    ),
)
```

Note that Syne Tune code is sent with the SageMaker Framework so that the `import syne_tune.report`
 that imports the reporter works when executing the training script, as such there is no need to install Syne Tune 
 in the docker image of the SageMaker Framework.

In addition, users can decide to run the tuning loop on a remote instance. This is helpful to avoid the need of letting 
a developer machine run and to benchmark many seed/model options.

```python
tuner = RemoteLauncher(
    tuner=Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        n_workers=n_workers,
        tuner_name="height-tuning",
        stop_criterion=StoppingCriterion(max_wallclock_time=600),
    ),
    # Extra arguments describing the ressource of the remote tuning instance and whether we want to wait
    # the tuning to finish. The instance-type where the tuning job runs can be different than the
    # instance-type used for evaluating the training jobs.
    instance_type='ml.m5.large',
)

tuner.run(wait=False)
```

In this case, the tuning loop is going to be executed on a `ml.m5.large` instance instead of running locally.
Both backends can be used when using the remote launcher (if you run with the Sagemaker backend the tuning loop 
will happen on the instance type specified in the remote launcher and the trials will be evaluated on the instance(s) 
configured in the SageMaker framework, this may include several instances in case of distributed training). 
In the case where the remote launcher is used with a SageMaker backend, a SageMaker job is 
created to execute the tuning loop which then schedule a new SageMaker training job for each configuration to be 
evaluated. The options and use-case in this table:

|Tuning loop | Trial execution | Use-case | example |
|------------|-----------------|----------|---------|
|Local	     | Local           | Quick tuning for cheap models, debugging.|	launch_height_baselines.py |
|Local	     | SageMaker	   | Avoid saturating machine with trial computation with expensive trial, possibly use distributed training, enable debugging the tuning loop on a local machine.	|launch_height_sagemaker.py |
|SageMaker   | Local	       | Run remotely to benchmark many HPO algo/seeds options, possibly with a big machine with multiple CPUs or GPUs.	|launch_height_sagemaker_remotely.py|
|SageMaker   | SageMaker	   | Run remotely to benchmark many HPO algo/seeds options, enable distributed training or heavy computation.	|launch_height_sagemaker_remotely.py with distribute_trials_on_SageMaker=True |

To summarize, to evaluate trial execution locally, users should use LocalBackend, to evaluate trials on SageMaker users should use the SageMakerBackend which allows to tune any SageMaker Estimator, see launch_height_baselines.py or launch_height_sagemaker.py for examples. To run a tuning loop remotely, RemoteLauncher can be used, see launch_height_sagemaker_remotely.py for an example.

**Output of a tuning job.** 

Every tuning experiment generates three files:
* `results.csv.zip` contains live information of all the results that were seen by the scheduler in addition to other information such as the decision taken by the scheduler, the wallclock time or the dollar-cost of the tuning (only on SageMaker). 
* `tuner.dill` contains the checkpoint of the tuner which include backend, scheduler and other information. This can be used to resume a tuning experiment, use Spot instance for tuning or perform fine-grain analysis of the scheduler state.
* `metadata.json` contains the time-stamp when the Tuner start to effectively run. It also contains possible user metadata information.

For instance, the following code:

```python
tuner = Tuner(
   trial_backend=trial_backend,
   scheduler=scheduler,
   n_workers=4,
   tuner_name="height-tuning",
   stop_criterion=StoppingCriterion(max_wallclock_time=600),
   metadata={'description': 'just an example'},
)
tuner.run()
```
runs a tuning by evaluating 4 configurations in parallel with a given backend/scheduler and stops after 600s.
Tuner appends a unique string to ensure unicity of tuner name (with the above example the id
 of the experiment may be `height-tuning-2021-07-02-10-04-37-233`). 
 Results are updated every 30 seconds by default which is configurable.

Experiment data can be retrieved at a later stage for further analysis with the following command:

```python
tuning_experiment = load_experiment("height-tuning-2021-07-02-10-04-37-233")
tuning_experiment = load_experiment(tuner.name) # equivalent
```

The results obtained load_experiment have the following schema.

```python
class ExperimentResult:
    name: str
    results: pandas.DataFrame
    metadata: Dict
    tuner: Tuner
```
Where metadata contains the metadata provided by the user (`{'description': 'just an example'} in this case) as well`
 as `st_tuner_creation_timestamp` which stores the time-stamp when the tuning actually started.

**Output of a tuning job when running tuning on SageMaker.**
When the tuning runs remotely on SageMaker, the results are stored at a regular
cadence to `s3://{s3_bucket}/syne-tune/{tuner-name}/`, where `s3_bucket`
can be configured (defaults to `default_bucket` of the session). For instance,
if the above experiment is run remotely, the following path is used for
checkpointing results and states:

`s3://sagemaker-us-west-2-{aws_account_id}/syne-tune/height-tuning-2021-07-02-10-04-37-233/results.csv.zip`

**Multiple GPUs.** If your instance has multiple GPUs, the local backend can run
different trials in parallel, each on its own GPU (with the option
`LocalBackend(rotate_gpus=True)`, which is activated by default). When a new
trial starts, it is assigned to a free GPU if possible. In case of ties, the
GPU with fewest prior assignments is chosen. If the number of workers is larger
than the number of GPUs, several trials will run as subprocesses on the same GPU.
If the number of workers is smaller or equal to the number of GPUs, each trial
occupies a GPU on its own, and trials can start without delay. Reasons to
choose `rotate_gpus=False` include insufficient GPU memory or the training
evaluation code making good use of multiple GPUs.


## Examples

Once you have a tuning script, you can call Tuner with any scheduler to perform your HPO.
You will find the following examples in [examples/](examples/) folder:
* [launch_height_baselines.py](examples/launch_height_baselines.py):
  launches HPO locally, tuning a simple script 
   [train_height_example.py](examples/training_scripts/height_example/train_height.py) for several baselines  
* [launch_height_ray.py](examples/launch_height_ray.py):
  launches HPO locally with [Ray Tune](https://docs.ray.io/en/master/tune/index.html)
  scheduler
* [launch_height_moasha.py](examples/launch_height_moasha.py):
  shows how to tune a script reporting multiple-objectives with multiobjective Asynchronous Hyperband (MOASHA)
* [launch_height_standalone_scheduler.py](examples/launch_height_standalone_scheduler.py):
  launches HPO locally with a custom scheduler that cuts any trial that is not
  in the top 80%
* [launch_height_sagemaker_remotely.py](examples/launch_height_sagemaker_remotely.py):
  launches the HPO loop on SageMaker rather than a local machine, trial can be executed either
  the remote machine or distributed again as separate SageMaker training jobs
* [launch_height_sagemaker.py](examples/launch_height_sagemaker.py):
  launches HPO on SageMaker to tune a SageMaker Pytorch estimator
* [launch_height_sagemaker_custom_image.py](examples/launch_height_sagemaker_custom_image.py):
  launches HPO on SageMaker to tune a entry point with a custom docker image
* [launch_plot_results.py](examples/launch_plot_results.py): shows how to plot
  results of a HPO experiment
* [launch_fashionmnist.py](examples/launch_fashionmnist.py):
launches HPO locally tuning a multi-layer perceptron on Fashion MNIST. This
employs an easy-to-use benchmark convention
* [launch_huggingface_classification.py](examples/launch_huggingface_classification.py):
  launches HPO on SageMaker to tune a SageMaker Hugging Face estimator for sentiment classification
* [launch_tuning_gluonts.py](examples/launch_tuning_gluonts.py):
  launches HPO locally to tune a gluon-ts time series forecasting algorithm
* [launch_rl_tuning.py](examples/launch_rl_tuning.py):
  launches HPO locally to tune a RL algorithm on the cartpole environment


## Running on SageMaker

If you want to launch experiments on SageMaker rather than on your local machine,
 you will need access to AWS and SageMaker on your machine. 
 
Make sure that:
* `awscli` is installed (see [this link](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html))
* `docker` is installed and running (see [this link](https://docs.docker.com/get-docker/))
* A SageMaker role have been created (see 
 [this page](https://docs.aws.amazon.com/glue/latest/dg/create-an-iam-role-sagemaker-notebook.html) for instructions if 
 you created a SageMaker notebook in the past, this role should have been created for you).
* AWS credentials have been set properly (see [this link](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html)).

Note: all those conditions are already met if you run in a SageMaker notebook, they are only relevant
if you run in your local machine or on another environment.
  
The following command should run without error if your credentials are available:
```bash
python -c "import boto3; print(boto3.client('sagemaker').list_training_jobs(MaxResults=1))"
```

Or run the following example that evaluates trials on SageMaker.
```bash
python examples/launch_height_sagemaker.py
```

Syne Tune allows you to launch HPO experiments remotely on SageMaker, instead of them
running on your local machine. This is particularly interesting for running many
experiments in parallel. Here is an example:
```bash
python examples/launch_height_sagemaker_remotely.py
```
If you run this for the first time, it will take a while, building a docker image with the
Syne Tune dependencies and pushing it to ECR. This has to be done only once, even if Syne
Tune source code is modified later on.

Assuming that `launch_height_sagemaker_remotely.py` is working for you now, you
should note that the script returns immediately after starting the experiment, which is
running as a SageMaker training job. This allows you to run many experiments in
parallel, possibly by using the [command line launcher](docs/command_line.md).

If running this example fails, you are probably not setup to build docker images and
push them to ECR on your local machine. Check that aws-cli is installed and that docker is running on your machine. 
After checking that those conditions are met (consider using a SageMaker notebook if not since AWS access and docker 
are configured automatically), you can try to building the image again by running with the following:
```bash
cd container
bash build_syne_tune_container.sh
```

To run on SageMaker, you can also use any custom docker images available on ECR.
See [launch_height_sagemaker_custom_image.py](examples/launch_height_sagemaker_custom_image.py)
for an example on how to run with a script with a custom docker image.


## Benchmarks

Syne Tune comes with a range of benchmarks for testing and demonstration.
[Turning your own tuning problem into a benchmark](docs/benchmarks.md) is simple
and comes with a number of advantages. As detailed in
[this tutorial](docs/command_line.md), you can use the CL launcher
[launch_hpo.py](benchmarking/cli/launch_hpo.py) in order to start one or more
experiments, adjusting many parameters of benchmark, back-end, tuner, or
scheduler from the command line. 

The simpler [benchmark_main.py](benchmarking/benchmark_loop/README.md) can also be used to
launch experiments that loops over many schedulers and benchmarks.

## Tutorials

Do you want to know more? Here are a number of tutorials.
* [Basics of Syne Tune](docs/tutorials/basics/README.md)
* [Using the built-in schedulers](docs/schedulers.md)
* [Choosing a configuration space](docs/search_space.md)
* [Using the command line launcher to benchmark schedulers](docs/command_line.md)
* [Using and extending the list of benchmarks](docs/benchmarks.md)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

