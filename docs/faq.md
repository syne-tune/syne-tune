# Syne Tune FAQ

## Table of contents

* [How can I run on AWS and SageMaker?](#running-on-sagemaker)
* [What are the metrics reported by default when calling the `Reporter`?](#reporter-metrics)
* [How can I utilize multiple GPUs?](#multiple-gpus)
* [What is the default mode when performing optimization?](#default-mode)
* [How are trials evaluated when evaluating trials on a local machine?](#trial-execution)
* [What does the output of the tuning contain?](#tuning-output)
* [Where can I find the output of the tuning?](#tuning-output-location)
* [How can I enable trial checkpointing?](#trial-checkpointing)
* [Which schedulers make use of checkpointing?](#schedulers-checkpointing)
* [Is the tuner checkpointed?](#tuner-checkpointing)
* [Where can I find the output of my trials?](#trial-output)
* [How can I plot the results of a tuning?](#plotting-tuning)
* [How can I specify additional tuning metadata?](#additional-metadata)
* [How do I append additional information to the results which are stored?](#logging-additional-information) 
* [I don’t want to wait, how can I launch the tuning on a remote machine?](#remote-tuning)
* [How can I run many experiments in parallel?](#experiment-parallel)
* [How can I access results after tuning remotely?](#results-remote-tuning)
* [How can I specify dependencies to remote launcher or when using the SageMaker backend?](#dependencies-remote)
* [How can I benchmark experiments from the command line?](#benchmark-cli)
* [What different schedulers do you support? What are the main differences between them?](#schedulers-supported)
* [How do I define the search space?](#search-space) 


### <a name="running-on-sagemaker"></a> How can I run on AWS and SageMaker?
If you want to launch experiments on SageMaker rather than on your local machine, you will need access to AWS and SageMaker on your machine.
Make sure that:

* `awscli` is installed (see [this link](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html))
* AWS credentials have been set properly (see [this link](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html)).
* A SageMaker role have been created (see [this page](https://docs.aws.amazon.com/glue/latest/dg/create-an-iam-role-sagemaker-notebook.html) for instructions if you created a SageMaker notebook in the past, this role should have been created for you)

The following command should run without error if your credentials are available:

```
python -c "import boto3; print(boto3.client('sagemaker').list_training_jobs(MaxResults=1))"
```

You can also run the following example that evaluates trials on SageMaker to test your setup.

```
python examples/launch_height_sagemaker.py
```

### <a name="reporter-metrics"></a> What are the metrics reported by default when calling the `Reporter`?

Whenever you call the reporter to log a result, the worker time-stamp, the worker time since the creation of the reporter and the number of times the reporter was called are logged under the fields `st_worker_timestamp`,  `st_worker_time`, and `st_worker_iter`. In addition, when running on SageMaker, a dollar-cost estimate is logged under the field `st_worker_cost`.

To see this behavior, you can simply call the reporter to see those metrics:

```
from syne_tune.report import Reporter
reporter = Reporter()
for step in range(3):
   reporter(step=step, metric=float(step) / 3)
   
# [tune-metric]: {"step": 0, "metric": 0.0, "st_worker_timestamp": 1644311849.6071281, "st_worker_time": 0.0001048670000045604, "st_worker_iter": 0}
# [tune-metric]: {"step": 1, "metric": 0.3333333333333333, "st_worker_timestamp": 1644311849.6071832, "st_worker_time": 0.00015910100000837701, "st_worker_iter": 1}
# [tune-metric]: {"step": 2, "metric": 0.6666666666666666, "st_worker_timestamp": 1644311849.60733, "st_worker_time": 0.00030723599996917983, "st_worker_iter": 2}
```

### <a name="multiple-gpus"></a> How can I utilize multiple GPUs?

To utilize multiple GPUs you can either use the backend `LocalBackend` which will run on the GPU available in a local machine. You can also run on a remote AWS machine with multiple GPUs using the local backend and the remote launcher, see [I don’t want to wait, how can I launch the tuning on a remote machine?]((#remote-tuning)) or run with the SageMaker backend which spins-up one training job per trial.

When evaluating trials on a local machine with `LocalBackend`, by default each trial is allocated to the least occupied GPU by setting `CUDA_VISIBLE_DEVICES` environment variable. 

### <a name="default-mode"></a> What is the default mode when performing optimization?

The default mode is min when performing optimization so the target metric is minimized. The mode can be configured when instantiating a scheduler.

### <a name="trial-execution"></a> How are trials evaluated when evaluating trials on a local machine?

When trials are executed locally (e.g. when `LocalBackend` is used), each trial is evaluated as a different sub-process.
As such the number of concurrent configurations evaluated at the same time (set by `n_workers`) should
account for the capacity of the machine where the trials are executed.

### <a name="tuning-output-location"></a> Where can I find the output of the tuning?

When running locally, the output of the tuning is saved under `~/syne-tune/{tuner-name}/`. 
When running remotely, by default the tuning output is synced regularly to  `s3://{sagemaker-default-bucket}/syne-tune/{tuner-name}/`.
Note: If you run remote tuning via the CLI, the tuning output is synced to `s3://{sagemaker-default-bucket}/syne-tune/{experiment-name}/{tuner-name}/`, where `experiment-name` is the prefix of `tuner-name` without the datetime extension (in the example above, `experiment-name = 'train-height'`).


### <a name="tuning-output"></a> What does the output of the tuning contain?

Syne Tune stores the following files `metadata.json`, `results.csv.zip`, and `tuner.dill` which are respectively metadata of the tuning job, results obtained at each time-step and state of the tuner.

### <a name="trial-checkpointing"></a> How can I enable trial checkpointing?

Since trials may be paused and resumed (either by schedulers or when using spot-instances), the user has the possibility to checkpoint intermediate results to avoid starting computation from scratch. Model outputs and checkpoints must be written into a specific local path given by the command line argument `st_checkpoint_dir`. Saving/loading model checkpoint from this directory enables to save/load the state when the job is stopped/resumed (setting the folder correctly and uniquely per trial is the responsibility of the backend), see [checkpoint_example.py](https://github.com/awslabs/syne-tune/blob/main/examples/training_scripts/checkpoint_example/checkpoint_example.py) to see a fully working example of a tuning script with checkpoint enabled.

When using SageMaker backend or tuning remotely, we use [SageMaker checkpoint mechanism](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html) under the hood to sync local checkpoints to s3. Checkpoints are synced to  `s3://{sagemaker-default-bucket}/syne-tune/{tuner-name}/{trial-id}/`, where `sagemaker-default-bucket` is the default bucket for SageMaker. 
This can also be configured, for instance if you launch experiments with the benchmarking CLI,
the files would be writen to `s3://{sagemaker-default-bucket}/syne-tune/{experiment-name}/{tuner-name}/{trial-id}/`.

### <a name="schedulers-checkpointing"></a> Which schedulers make use of checkpointing?

Checkpointing means storing the state of a trial (i.e., model parameters, optimizer or learning rate scheduler 
parameters), so that it can be paused and potentially resumed at a later point in time, without having to start 
training from scratch. Syne Tune checkpointing support is detailed in 
[checkpointing](https://github.com/awslabs/syne-tune/blob/main/docs/benchmarks.md#checkpointing) and 
[checkpoint_example.py](https://github.com/awslabs/syne-tune/blob/main/examples/training_scripts/checkpoint_example/checkpoint_example.py). Once checkpointing to and from a local file is added to a training script, Syne Tune is managing the checkpoints (e.g., copy to/from cloud storage for distributed back-ends).

The following schedulers make use of checkpointing:

* Promotion-based Hyperband: `HyperbandScheduler(type='promotion', ...)`
    The code runs without checkpointing, but in this case, any trial which is resumed is started from scratch. For example, if a trial was paused after 9 epochs of training and is resumed later, training starts from scratch and the first 9 epochs are wasted effort. Moreover, extra variance is introduced by starting from scratch, since weights may be initialized differently. It is not recommended to run promotion-based Hyperband without checkpointing.
* Population-based training: `PopulationBasedTraining`
    PBT does not work without checkpointing.
* Synchronous Hyperband: `SynchronousGeometricHyperbandScheduler`
    This code runs without checkpointing, but wastes effort in the same sense as promotion-based asynchronous Hyperband

### <a name="tuner-checkpointing"></a> Is the tuner checkpointed?

Yes. When performing the tuning, the tuner state is regularly saved on the experiment path under `tuner.dill`
(every 10 seconds which can be configured with `results_update_interval`).
This allows to use spot-instances when running a tunning remotely with the remote launcher. It also allows to 
resume a past experiment or analyse the state of scheduler at any point.

### <a name="trial-output"></a> Where can I find the output of my trials?

When running  `LocalBackend` locally, results of trials are saved under `~/syne-tune/{tuner-name}/{trial-id}/` and contains the following files:

* config.json: configuration that is being evaluated in the trial
* std.err: standard error 
* std.out: standard output

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

When running tuning remotely with the remote launcher, only `metadata.json`, `results.csv.zip` and `tuner.dill` 
are synced with S3 unless `store_logs_localbackend` in which case the trial logs and informations are also persisted.

### <a name="plotting-tuning"></a> How can I plot the results of a tuning?

The easiest way to plot the result of a tuning experiment is to call the following:

```
tuner = Tuner(
    ...
    tuner_name="plot-results-demo",
)
tuner.run()
tuning_experiment = load_experiment(tuner.name)
tuning_experiment.plot()
```

This generates a plot of the best value found over time. Note that this you can also plot the results while the experiment is running as results are updated continuously.

### <a name="additional-metadata"></a> How can I specify additional tuning metadata?

By default, Syne Tune stores the time, the names and modes of the metric being tuner, the name of the entrypoint, the name backend and the scheduler name.
You can also add custom metadata to your tuning job by setting `metadata` in `Tuner` as follow:

```
tuner = Tuner(
    ...
    tuner_name="plot-results-demo",
    metadata={"tag": "special-tag", "user": "alice"},
)
```

All Syne Tune and user metadata are saved when the tuner starts under `metadata.json`.


### <a name="logging-additional-information"></a> How do I append additional information to the results which are stored? 

Results are processed and stored by callbacks passed to `Tuner`, in particular see 
[tuner_callback.py](https://github.com/awslabs/syne-tune/blob/main/syne_tune/tuner_callback.py#L80). 
In order to add more information to these results, you can inherit from `StoreResultsCallback`. 
A good example is given in 
[searcher_callback.py](https://github.com/awslabs/syne-tune/blob/main/syne_tune/optimizer/schedulers/searchers/searcher_callback.py#L63).

If you run experiments with tabulated benchmarks using the `SimulatorBackend`, as demonstrated in 
[launch_nasbench201_simulated.py](https://github.com/awslabs/syne-tune/blob/main/examples/launch_nasbench201_simulated.py), results are stored by `SimulatorCallback` instead, and you need to inherit from this class, as shown in 
[searcher_callback.py](https://github.com/awslabs/syne-tune/blob/main/syne_tune/optimizer/schedulers/searchers/searcher_callback.py#L88).

### <a name="remote-tuning"></a> I don’t want to wait, how can I launch the tuning on a remote machine?

You can use the remote launcher to launch a tuning on a remote machine. The remote launcher supports both `LocalBackend` and `SagemakerBackend`. In the former case, multiple trials will be evaluated on the remote machine (one use-case being to use a beefy machine), in the latter case trials will be evaluated as separate SageMaker training jobs.

See for an example on how to run tuning with the remote launcher: [launch_height_sagemaker_remotely.py](https://github.com/awslabs/syne-tune/blob/main/examples/launch_height_sagemaker_remotely.py)

### <a name="experiment-parallel"></a> How can I run many experiments in parallel?

You can call the remote launcher multiple times to schedules a list of experiments. In some case, you will want more flexibility and directly write your experiment loop, you can check 
[benchmark_loop](https://github.com/awslabs/syne-tune/tree/main/benchmarking/benchmark_loop) for an example.

### <a name="results-remote-tuning"></a> How can I access results after tuning remotely?

You can either call `load_experiment("{tuner-name}")` which will download files from s3 if the experiment is not found locally. You can also sync directly files from s3 under `~/syne-tune/` folder in batch for instance by running:

```
aws s3 sync s3://{sagemaker-default-bucket}/syne-tune/{tuner-name}/ ~/syne-tune/  --include "*"  --exclude "*tuner.dill"
```

To get all results without the tuner state (you can ommit the include and exclude if you also want to include the tuner state).

### <a name="dependencies-remote"></a> How can I specify dependencies to remote launcher or when using the SageMaker backend?

When you run remote code, you often need to install packages (e.g. scipy) or have custom code available.

* To install packages, you can add a file `requirements.txt` in the same folder as your endpoint script. All those packages will be installed by SageMaker when docker container starts.
* To include custom code (for instance a library that you are working on), you can set the parameter `dependencies` on the remote launcher or on a SageMaker framework to a list of folders. The folders indicated will be compressed, sent to s3 and added to the python path when the container starts. You can see [launch_remote.py](https://github.com/awslabs/syne-tune/blob/main/benchmarking/benchmark_loop/launch_remote.py#L28) for an example setting dependencies in a SageMaker estimator.

### <a name="benchmark-cli"></a> How can I benchmark different methods?

Syne Tune provides a command line launcher in `benchmarking/cli/launch_hpo.py`. It allows to launch a range of experiments with a single command. The CLI is documented at https://github.com/awslabs/syne-tune/blob/main/docs/command_line.md. In order to use the CLI with your own training script, you need add some meta-information, as explained in https://github.com/awslabs/syne-tune/blob/main/docs/benchmarks.md.

Another possibility is to write a custom loop over different methods and benchmarks and an example is provided 
and documented in this [directory](benchmarking/benchmark_loop/README.md).




### <a name="schedulers-supported"></a> What different schedulers do you support? What are the main differences between them?

We refer to HPO algorithms as *schedulers*. A scheduler decides which configurations to assign to new trials, but also when to stop a running or resume a paused trial. Some schedulers delegate the first decision to a *searcher*. The most important differences between schedulers in the single-objective case are:

* Does the scheduler stop trials early or pause and resume trials (`HyperbandScheduler`) or not (`FIFOScheduler`). The former requires a resource dimension (e.g., number of epochs; size of training set) and slightly more elaborate reporting (e.g., evaluation after every epoch), but can outperform the latter by a large margin.
* Does the searcher suggest new configurations by uniform random sampling (`searcher='random'`) or by sequential model-based decision-making (`searcher='bayesopt'`, `searcher='kde'`). The latter can be more expensive if a lot of trials are run, but can also be more sample-efficient.

An overview of this landscape is given in https://github.com/awslabs/syne-tune/blob/main/docs/schedulers.md. Further schedulers provided by Syne Tune include:

* Population based training (PBT): https://github.com/awslabs/syne-tune/blob/main/examples/launch_pbt.py
* Multi-objective asynchronous successive halving (MOASHA): https://github.com/awslabs/syne-tune/blob/main/examples/launch_moasha_instance_tuning.py
* Constrained Bayesian optimization: https://github.com/awslabs/syne-tune/blob/main/examples/launch_bayesopt_constrained.py
* Bayesian optimization by density-ratio estimation (BORE): https://github.com/awslabs/syne-tune/blob/main/syne_tune/optimizer/baselines.py#L75
* Regularized evolution: https://github.com/awslabs/syne-tune/blob/main/syne_tune/optimizer/baselines.py#L87
* Median stopping rule: https://github.com/awslabs/syne-tune/blob/main/syne_tune/optimizer/schedulers/median_stopping_rule.py
* Synchronous Hyperband: https://github.com/awslabs/syne-tune/blob/main/syne_tune/optimizer/schedulers/synchronous/hyperband_impl.py#L42
* Transfer learning schedulers: https://github.com/awslabs/syne-tune/blob/main/examples/launch_nas201_transfer_learning.py
* Wrappers for Ray Tune schedulers: https://github.com/awslabs/syne-tune/blob/main/examples/launch_height_ray.py

Most of those methods can be accessed with short names by from [baselines.py](https://github.com/awslabs/syne-tune/blob/main/syne_tune/optimizer/baselines.py).

### <a name="search-space"></a> How do I define the search space? 

While the training script defines the function to be optimized, some care needs to be taken to define the search space for the hyperparameter optimization problem. This being a global optimization problem without gradients easily available, it is most important to reduce the number of parameters. Some advice is given in https://github.com/awslabs/syne-tune/blob/main/docs/search_space.md.

A powerful approach is to run experiments in parallel. Namely, split your hyperparameters into groups A, B, such that HPO over B is tractable. Draw a set of N configurations from A at random, then start N HPO experiments in parallel, where in each of them the search space is over B only, while the parameters in A are fixed. Syne Tune supports massively parallel experimentation (see https://github.com/awslabs/syne-tune/blob/main/docs/command_line.md#launching-many-experiments).