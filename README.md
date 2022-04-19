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
* `kde`: For KDE optimizer

For instance, `pip install 'syne-tune[gpsearchers]'` will install Syne Tune along with many built-in Gaussian process 
optimizers.

To install the latest version from git, run the following:

```bash
pip install git+https://github.com/awslabs/syne-tune.git
```

For local development, we recommend to use the following setup which will enable you to easily test your changes: 

```bash
pip install --upgrade pip
git clone https://github.com/awslabs/syne-tune.git
cd syne-tune
pip install -e '.[extra]'
```

To run unit tests, simply run `pytest` in the root of this repository.

To run all tests whose name begins with `test_async_scheduler`, you can use the following
```bash
pytest -k test_async_scheduler
```


## Getting started

To enable tuning, you have to report metrics from a training script so that they can be communicated later to Syne Tune,
this can be accomplished by just calling `report(epoch=epoch, loss=loss)` as shown in the example bellow:

```python
# train_height.py
import logging
import time

from syne_tune import Reporter
from argparse import ArgumentParser

if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('--steps', type=int)
    parser.add_argument('--width', type=float)
    parser.add_argument('--height', type=float)

    args, _ = parser.parse_known_args()
    report = Reporter()

    for step in range(args.steps):
        dummy_score = (0.1 + args.width * step / 100) ** (-1) + args.height * 0.1
        # Feed the score back to Syne Tune.
        report(step=step, mean_loss=dummy_score, epoch=step + 1)
        time.sleep(0.1)
```

Once you have a script reporting metric, you can launch a tuning as-follow:

```python
from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint
from syne_tune.optimizer.baselines import ASHA

# hyperparameter search space to consider
config_space = {
    'steps': 100,
    'width': randint(1, 20),
    'height': randint(1, 20),
}

tuner = Tuner(
    trial_backend=LocalBackend(entry_point='train_height.py'),
    scheduler=ASHA(
        config_space, metric='mean_loss', resource_attr='epoch', max_t=100,
        search_options={'debug_log': False},
    ),
    stop_criterion=StoppingCriterion(max_wallclock_time=15),
    n_workers=4,  # how many trials are evaluated in parallel
)
tuner.run()
```

The above example runs ASHA with 4 asynchronous workers on a local machine.

## Examples

You will find the following examples in [examples/](examples/) folder illustrating different functionalities provided
by Syne Tune:
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

## FAQ and Tutorials

You can check our [FAQ](docs/faq.md), to learn more about Syne Tune functionalities. 

* [How can I run on AWS and SageMaker?](docs/faq.md)
* [What are the metrics reported by default when calling the `Reporter`?](docs/faq.md)
* [How can I utilize multiple GPUs?](docs/faq.md)
* [What is the default mode when performing optimization?](docs/faq.md)
* [How are trials evaluated when evaluating trials on a local machine?](docs/faq.md)
* [What does the the output of the tuning contain?](docs/faq.md)
* [How can I enable trial checkpointing?](docs/faq.md)
* [Which schedulers make use of checkpointing?](docs/faq.md)
* [Is the tuner checkpointed?](docs/faq.md)
* [Where can I find the output of my trials?](docs/faq.md)
* [Where can I find the output of the tuning?](docs/faq.md)
* [How can I plot the results of a tuning?](docs/faq.md)
* [How can I specify additional tuning metadata?](docs/faq.md)
* [How do I append additional information to the results which are stored?](docs/faq.md) 
* [I don’t want to wait, how can I launch the tuning on a remote machine?](docs/faq.md)
* [How can I run many experiments in parallel?](docs/faq.md)
* [How can I access results after tuning remotely?](docs/faq.md)
* [How can I specify dependencies to remote launcher or when using the SageMaker backend?](docs/faq.md)
* [How can I benchmark experiments from the command line?](docs/faq.md)
* [What different schedulers do you support? What are the main differences between them?](docs/faq.md)
* [How do I define the search space?](docs/faq.md) 

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

