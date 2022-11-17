# Benchmarking with SageMaker Back-end

The SageMaker back-end allows you to run distributed tuning across several
instances, where the number of parallel evaluations is not limited by the
configuration of an instance, but only by your compute budget.


## Defining the Experiment

The scripts required to define an experiment are pretty much the same as in
the local back-end case. We will look at an example in
[benchmarking/nursery/launch_sagemaker](../../../benchmarking/nursery/launch_sagemaker/).
Common code used in these benchmarks can be found in
[benchmarking/commons](../../../benchmarking/commons/):
* Local launcher [hpo_main_sagemaker.py](../../../benchmarking/commons/hpo_main_sagemaker.py)
* Remote launcher [launch_remote_sagemaker.py](../../../benchmarking/commons/launch_remote_sagemaker.py)
* Benchmark definitions [benchmark_definitions](../../../benchmarking/commons/benchmark_definitions/__init__.py)

The scripts
[baselines.py](../../../benchmarking/nursery/launch_sagemaker/baselines.py),
[hpo_main.py](../../../benchmarking/nursery/launch_sagemaker/hpo_main.py),
[launch_remote.py](../../../benchmarking/nursery/launch_sagemaker/launch_remote.py)
are identical in structure to what happens in the
[local back-end case](bm_local.md#defining-the-experiment), with the only difference
that [hpo_main_sagemaker.py](../../../benchmarking/commons/hpo_main_sagemaker.py)
or [launch_remote_sagemaker.py](../../../benchmarking/commons/launch_remote_sagemaker.py)
are imported from. Moreover, Syne Tune dependencies need to be specified in
[requirements.txt](../../../benchmarking/nursery/launch_sagemaker/requirements.txt).

In terms of benchmarks, the same definitions can be used for the SageMaker back-end,
in particular you can select from
[real_benchmark_definitions](../../../benchmarking/commons/benchmark_definitions/real_benchmark_definitions.py).
However, the functions there are called with `sagemaker_backend=True`, which can
lead to different values in
[RealBenchmarkDefinition](../../../benchmarking/commons/benchmark_definitions/common.py#L56).
For example,
[resnet_cifar10_benchmark](../../../benchmarking/commons/benchmark_definitions/resnet_cifar10.py#L23)
returns `instance_type=ml.g4dn.xlarge` for the SageMaker back-end (1 GPU per
instance), but `instance_type=ml.g4dn.12xlarge` for the local back-end (4 GPUs per
instance). This is because for the local back-end to support `n_workers=4`, the
instance needs to have at least 4 GPUs, but for the SageMaker back-end, each
worker uses its own instance, so a cheaper instance type can be used.


## Launching Experiments Locally

Here is an example of how experiments with the SageMaker back-end are launched
locally:

```bash
python benchmarking/nursery/launch_sagemaker/hpo_main.py \
  --experiment_tag tutorial_sagemaker --benchmark resnet_cifar10 \
  --method ASHA --num_seeds 1
```

This call launches a single experiment on the local machine (however, each
trial launches the training script as a SageMaker training job, using the
instance type suggested for the benchmark). The command line arguments are
the same as in the [local back-end case](bm_local.md#launching-experiments-locally).
Additional arguments are:
* `n_workers`, `max_wallclock_time`: Overwrite the default values
   for the selected benchmark.
* `max_failures`: Number of trials which can fail without terminating the
  entire experiment.
* `warm_pool`: This flag is discussed [below](bm_sagemaker.md#using-sagemaker-managed-warm-pools).

If you defined additional arguments via `extra_args`, you can use them
here as well.


## Launching Experiments Remotely

Sagemaker back-end experiments can also be launched remotely, in which case
each experiment is run in a SageMaker training job, using a cheap instance
type, within which trials are executed as SageMaker training jobs as well.
The usage is the same as in the
[local back-end case](bm_local.md#launching-experiments-remotely).

However, at present, there are still some glitches which affect remote
launching of SageMaker back-end experiments. As long as these are not
resolved, this feature remains experimental.


## Using SageMaker Managed Warm Pools

The SageMaker back-end supports
[SageMaker managed warm pools](https://docs.aws.amazon.com/sagemaker/latest/dg/train-warm-pools.html),
a recently launched feature of SageMaker. In a nutshell, this feature allows
customers to circumvent start-up delays for SageMaker training jobs which share
a similar configuration (e.g., framework) with earlier jobs which have already
terminated. For Syne Tune with the SageMaker back-end, this translates to
experiments running faster or, for a fixed `max_wallclock_time`, running more
trials. Warm pools are used if the command line argument `--warm_pool 1` is
used with `hpo_main.py`. For the example above:

```bash
python benchmarking/nursery/launch_sagemaker/hpo_main.py \
  --experiment_tag tutorial_sagemaker --benchmark resnet_cifar10 \
  --method ASHA --num_seeds 1 --warm_pool 1
```

The warm pool feature is most useful with multi-fidelity HPO methods (such as
`ASHA` and `MOBSTER` in our example). Some points you should be aware of:
* Warm pools are a billable resource, and you may incur extra costs. You have
  to request warm pool quota increases for instance types you would like to use.
  For our example, you need to have quotas for (at least) four `ml.g4dn.xlarge`
  instances, **both** for training and warm pool usage.
* As of today, warm pools decrease start-up times, but stopping a SageMaker
  training job still has a sizable delay. It is therefore strongly recommended
  to use `type="promotion"` with the asynchronous HPO methods in Syne Tune
  (e.g., `ASHA`, `MOBSTER`, `HyperTune`), as they do not use stop signals (the
  default is `type="stopping"`).
  The synchronous multi-fidelity methods (e.g., `SyncHyperband`, `SyncBOHB`,
  `DEHB`, `SyncMOBSTER`) don't do this either.
* As a sanity check, you can watch the training jobs in the console. You should
  see `InUse` and `Reused` in the *Warm pool status* column. Running the
  example above, the first 4 jobs should complete in about 7 to 8 minutes,
  while all subsequent jobs should take only 2 to 3 minutes.


In the [next section](bm_contributing.md), you will learn how to contribute
your own benchmark to Syne Tune.
