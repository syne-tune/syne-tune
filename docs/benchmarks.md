# Syne Tune: Using and Extending Benchmarks


## Why Benchmarks?

The most flexible way to make your training code ready for tuning and to run
an HPO experiments is with Python scripts. However, when you end up working
with a model for an extended period and would like to rapidly compare different
options, it pays off to use benchmarks:
* Once your tuning problem is a benchmark, you can run experiments from the
  [command line launcher](command_line.md).
  You can store results of different experiments, or random repeats of an
  experiment to S3 and create comparative plots.
* If you work on a range of problems in a team, a benchmark can be a clean
  way of specifying variations of a tuning problem, and to collect results
  and analyses over time.
* Turning your annotated training code into a benchmark requires only minimal
  extra effort.

A number of benchmarks are included, they are indexed in
[benchmark_factory.py](../benchmarking/cli/benchmark_factory.py)


## Including your own benchmark

Please have a look at
[mlp_on_fashion_mnist.py](../benchmarking/training_scripts/mlp_on_fashion_mnist/mlp_on_fashion_mnist.py)
and [definition_mlp_on_fashion_mnist.py](../benchmarking/definitions/definition_mlp_on_fashion_mnist.py).
The first is a pure training script, of the type you already know. The second
is a benchmark definition, containing meta-data to complete the definition of a
benchmark. Before moving on, we note that a benchmark can have different
variants, selected by parameters. Also, the same training script can be used by
several benchmarks or benchmark variants.

Here is what a benchmark definition needs to contain:
* `_config_space`: In fact, the config space is often defined in the training
  script, because it determines its command line arguments. It is meant to be a
  default, which can still be overwritten by the user
  (see [examples/launch_fashionmnist.py](../examples/launch_fashionmnist.py)).
* `mlp_fashionmnist_benchmark` returns a dict with all required information and
  defaults for the benchmark. You need to specify the names of metrics (keys used
  in the `report` call). `mode` is "max" or "min", depending on whether `metric`
  is to be maximized or minimized. If your scripts requires parameters besides
  tunable hyperparameters, `config_space` needs to contain values for them. This
  is why we construct the complete `config_space` from `_config_space` (which
  defines hyperparameters only).
* `mlp_fashionmnist_default_params` returns a dict of benchmark-specific default
  values for scheduler or back-end parameters (see below). If the user does not
  specify values, these default values have priority over global default
  values. Some entries (e.g., `max_resource_level`, `dataset_path`,
  `report_current_best`) can parameterize the benchmark itself. Importantly,
  all keys in `default_params` returned here are available as command line
  arguments, even if they are specific to the particular benchmark (for
  example, `report_current_best` features in some benchmarks, but not in all).
* Optionally, a cost model can be defined, which is required by some schedulers.
  The cost model to be used is returned as `cost_model` entry of the dict
  returned by `mlp_fashionmnist_benchmark`.

Let us have a closer look at the dict returned by `mlp_fashionmnist_benchmark`:
* `script` and `config_space`: These are mandatory. `script` points to the
  training code, `config_space` is the configuration space extended by constant
  attributes.
* `metric`, `mode`, `resource_attr`, `elapsed_time_attr`: Names of metrics
  reported by the training script (except for `mode`, which specifies whether
  `metric` is minimized or maximized). `metric` is mandatory. If you report after
  every epoch, `resource_attr` is the name of the resource level (i.e., epoch
  number). `elapsed_time_attr` is the time since start of the training script
  (possibly except overhead for downloading data).
* `max_resource_attr`: If your script consists of a loop over resource levels
  (e.g., epochs), it should have an argument for the maximum resource level
  (e.g., `epochs` in
  `mlp_on_fashionmnist`). `max_resource_attr` contains the name of this
  argument. This (optional) information is used for several purposes. First,
  for schedulers which need to know the maximum epoch, they can infer it from
  the search space (since they know the key name). Second, schedulers which pause
  and promote trials can use this information to set the maximum epoch parameter 
  in the config of a job to be executed. This saves overhead, since the job
  need not be stopped by the back-end, but terminates on its own. It is strongly
  recommended to specify this parameter.

Once you coded up your own benchmark, make sure to link into
[benchmark_factory.py](../benchmarking/cli/benchmark_factory.py), providing a
new name.

### Specifying Default Parameters

The `dict` returned by `mlp_fashionmnist_default_params` is used to specify
default values for scheduler or back-end parameters. If the user does not
specify values, these default values have priority over global defaults. The
keys in this `dict` correspond to command line argument names, which in
most cases are the same as argument names to scheduler or back-end
constructors. Other entries in the dict can parameterize the benchmark itself,
via constant entries in `config_space`. Importantly, all keys in the `dict`
returned here are also available as command line arguments, even if they do
not appear in other benchmarks. Some notable entries of the `default_params`
dictionary are:
* `max_resource_level`: Most benchmarks iterate over resource levels (e.g.,
  epochs, batches, or subset fractions), and they report intermediate results
  at the end of each resource level. `max_resource_level` is the default
  value for the maximum resource level. Resource levels are positive integers,
  starting with 1, and most benchmarks report at all levels between 1 and
  `max_resource_level`.
* `points_to_evaluate`: Allows to specify a list of configurations which are
  evaluated first. If your training code corresponds to some open source ML
  algorithm, you may want to use the defaults provided in the code. More details
  are given [here](schedulers.md#fifoscheduler).
* `instance_type`: Choose a default instance type for which your benchmark
  runs most economically (used with `sagemaker` back-end or remote tuning, see
  [here](command_line.md)).
* `num_workers`: Default for number of workers, i.e. the maximum number of
  parallel evaluations. The exact number of evaluations running in parallel
  may change with time, and is bounded by the instance limits (e.g., number
  of GPUs or vCPUs).
* `framework`: Selects SageMaker framework which covers the essential
  dependencies of the training scripts (more details below).
* `framework_version`: Version of SageMaker framework to be used (more details
  below).


### SageMaker-specific Aspects

We have already seen that the training evaluation script has the form of a
endpoint script passed to a SageMaker training job. In general, the code will
have non-trivial dependencies (e.g., PyTorch, TensorFlow, Hugging Face,
scikit-learn). SageMaker provides different ways for customers to manage these
dependencies, which will have to be installed and setup automatically on an
instance of chosen type before the code can be executed.

If you use the SageMaker back-end (`--backend sagemaker` in the
[command line launcher](command_line.md)), you can make use of
[SageMaker frameworks](https://sagemaker.readthedocs.io/en/stable/frameworks/index.html).
The benchmarking mechanism of Syne Tune currently supports the following
frameworks:

* PyTorch
* HuggingFace

Others will be added on demand. If a benchmark uses a framework for its
dependencies, this must be specified in `default_params['framework']`, the
benchmark version can be set in `default_params['framework_version']`.

Your benchmark code may mostly map to a framework, but come with additional
minor dependencies not contained there. Such dependencies can be listed in a
file `dependencies.txt` in the same directory as the training evaluation
script.

If this route does not work for your particular benchmark, or if you do not
want to use a SageMaker framework, you need to build a Docker image for your
dependencies. Once
this is done and uploaded to ECR, its URI must be passed to the back-end. If
you use the CLI, the argument is `image_uri`. If you plan to use such a
benchmark on your own, you can specify the URI in `default_params['image_uri']`.
Note that the Docker image does not have to contain the code for your
benchmark, which is copied by SageMaker separately. In particular, you can
change your source code without having to rebuild the image.

If you use the local back-end (the default in the command line launcher),
Syne Tune currently does not support SageMaker frameworks, and the
settings in `default_params['framework']` and `default_params['framework_version']`
are ignored. As explained in detail in the [command line launcher](command_line.md)
tutorial, you can still run your experiments as SageMaker training jobs (a feature
called *remote tuning*), but these training jobs use the PyTorch framework. If
you need additional dependencies, you need to specify them in `dependencies.txt`.


## Checkpointing

You may have noticed `resume_from_checkpointed_model`,
`checkpoint_model_at_rung_level`, `add_checkpointing_to_argparse`,
`pytorch_load_save_functions` in the example above. These help with
**checkpointing** in benchmarks. Some schedulers can pause trials at a certain
resource level (i.e., number of epochs) and may resume any paused trial at a
later stage. In order to be competitive, such schedulers require checkpointing:
the mutable state of the evaluation (e.g., model weights, optimizer parameters)
is stored when the trial is paused, and is loaded once a trial resumes. Without
checkpointing support, training for a resumed trial has to start from scratch,
which is wasteful.

Let us look at code snippets from
[lstm_wikitext2.py](../benchmarking/nursery/lstm_wikitext2/lstm_wikitext2.py):

```python
# ...

from benchmarking.utils import resume_from_checkpointed_model, \
    checkpoint_model_at_rung_level, add_checkpointing_to_argparse, \
    pytorch_load_save_functions
# ...

# Objective to tune
def objective(config):
    # ...

    # Model to train:
    model = TransformerModel(
        ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    # ...

    # Checkpointing
    # Note that `lr` and `best_val_loss` are also part of the state to be
    # checkpointed. In order for things to work out, we keep them in a
    # dict (otherwise, they'd not be mutable in `load_model_fn`,
    # `save_model_fn`.
    mutable_state = {
        'lr': config['lr'],
        'best_val_loss': None}

    # `pytorch_load_save_functions` is a helper to create `load_model_fn`,
    # `save_model_fn`. It accepts a dict of PyTorch objects which implement
    # `load_state_dict` and `state_dict`, as well as (optional) a dict with
    # standard value types (`mutable_state` here)
    load_model_fn, save_model_fn = pytorch_load_save_functions(
        {'model': model}, mutable_state)

    # Resume from checkpoint (optional)
    # [2]:
    resume_from = resume_from_checkpointed_model(config, load_model_fn)

    # Loop over epochs.
    for epoch in range(resume_from + 1, config['epochs'] + 1):
        train(model, corpus, criterion, train_data, mutable_state['lr'],
              batch_size, clip)
        val_loss = evaluate(model, corpus, criterion, val_data)
        # ...
        
        best_val_loss = mutable_state['best_val_loss']
        if not best_val_loss or val_loss < best_val_loss:
            best_val_loss = val_loss
            mutable_state['best_val_loss'] = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            mutable_state['lr'] /= config['lr_factor']

        # Feed the score back back to Tune.
        _loss = best_val_loss if report_current_best else val_loss
        objective = -math.exp(_loss)
        report(
            epoch=epoch,
            objective=objective)

        # Write checkpoint (optional)
        # [1]:
        checkpoint_model_at_rung_level(config, save_model_fn, epoch)


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True)
    # ...
    # [3]:
    add_checkpointing_to_argparse(parser)

    args, _ = parser.parse_known_args()
    objective(config=vars(args))
```

In order to support checkpointing for "pause and resume" schedulers, a
script has to be extended in two places:
* Checkpoints have to be written at the end of certain epochs (namely those
  after which the scheduler may pause the trial). This is dealt with by
  `checkpoint_model_at_rung_level(config, save_model_fn, epoch)` at [1]. Here,
  `epoch` is the current epoch, allowing the function to decide whether to
  checkpoint or not. `save_model_fn` stores the current mutable state
  along with `epoch` to a local path (see below). Finally, `config` contains
  arguments provided by the scheduler (see below).
* Before the training loop starts (and optionally), the mutable state to start
  from has to be loaded from a checkpoint. This is done by
  `resume_from_checkpointed_model(config, load_model_fn)` in [2]. If the
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
`config`. You don't have to worry about this: `add_checkpointing_to_argparse(parser)`
in [3] adds corresponding arguments to the parser.

### How does Checkpointing Work Internally?

Internally, the training scripts stores and loads checkpoints to and from a
local directory, the name of which is provided via `config`. The back-end makes
sure checkpoint files are synced to instance-independent storage, using unique
file names. Checkpointing is activated only for schedulers supporting pause and
resume. In this case:
* While a trial is running, checkpoints are stored to a local directory at the
  end of each epoch. Here, newer files overwrite older ones.
* Once the scheduler pauses a trial, files in the local checkpoint directory
  are synced to a unique location.
* When the scheduler decides to resume a trial, it first loads the corresponding
  checkpoint file(s) from permanent storage to the local checkpoint directory
  of the worker instance which will execute the trial. The training script can
  then resume from this checkpoint.
