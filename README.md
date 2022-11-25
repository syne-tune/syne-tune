# Syne Tune: Large-Scale and Reproducible Hyperparameter Optimization

[![Release](https://img.shields.io/badge/release-0.3-brightgreen.svg)](https://pypi.org/project/syne-tune/)
[![Python Version](https://img.shields.io/badge/3.7%20%7C%203.8%20%7C%203.9-brightgreen.svg)](https://pypi.org/project/syne-tune/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://pepy.tech/badge/syne-tune/month)](https://pepy.tech/project/syne-tune)

![Alt Text](docs/synetune.gif)

This package provides state-of-the-art algorithms for hyperparameter optimization (HPO) with the following key features:
* Wide coverage (>20) of different HPO methods, including:
  * Asynchronous versions to maximize utilization and distributed versions (i.e., with multiple workers);
  * Multi-fidelity methods supporting model-based decisions (BOHB and MOBSTER);
  * Hyperparameter transfer learning to speed up (repeated) tuning jobs;
  * Multi-objective optimizers that can tune multiple objectives simultaneously (such as accuracy and latency).
* HPO can be run in different environments (locally, AWS, simulation) by changing just one line of code.
* Out-of-the-box tabulated benchmarks that allows you simulate results in seconds while preserving the real dynamics of asynchronous or synchronous HPO with any number of workers.

## Installing

To install Syne Tune from pip, you can simply do:

```bash
pip install 'syne-tune[extra]'
```

or to get the latest version from git: 

```bash
pip install --upgrade pip
git clone https://github.com/awslabs/syne-tune.git
cd syne-tune
pip install -e '.[extra]'
```

When installing Syne Tune from sources, we recommend to use a virtual environment. You can see the FAQ [What are the different installations options supported?](docs/faq.md#installations) for more install options.

See our [change log](CHANGELOG.md) to see what changed in the latest version. 

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

## Supported HPO methods

The following hyperparameter optimization (HPO) methods are available in Syne Tune:

Method | Reference | Searcher | Asynchronous? | Multi-fidelity? | Transfer? | Checkpointing?
:--- | :---: | :---: | :---: | :---: | :---: | :---:
Grid Search | Someone (sometime) | non-random | no | no | no | dunno
Rand Search | Bergstra, et al. (2011) | random | yes | no | no | dunno
Median Stopping Rule | Golovin, et al. (2017) | any | yes | yes | no | dunno
Bayesian Optimization | Snoek, et al. (2012) | model-based | yes | no | no | dunno
TPE | Bergstra, et al. (2011) | model-based | yes | no | no | dunno
BORE | Tiao, et al. (2021) | model-based | yes | no | no | dunno
SyncHyperband | Li, et al. (2018) | random | no | yes | no | dunno
HyperTune | Li, et al. (2022) | model-based | yes | yes | no | dunno
BOHB | Falkner, et al. (2018) | model-based | yes | yes | no | dunno
SyncBOHB | Falkner, et al. (2018) | model-based | no | yes | no | dunno
ASHA | Li, et al. (2019) | random | yes | yes | no | dunno
MOBSTER | Klein, et al. (2020) | model-based | yes | yes | no | dunno
SyncMOBSTER | Klein, et al. (2020) | model-based | no | yes | no | dunno
REA | Real, et al. (2019) | ?? | yes | no | no | dunno
PBT | Jaderberg, et al. (2017) | random | no | yes | no | dunno
DEHB | Awad, et al. (2021) | random | no | yes | no | dunno
PASHA | Bohdal, et al. (2022)| random | yes | yes | no | dunno
ZeroShotTransfer | Wistuba, et al. (2015) | non-random | yes | no | yes | dunno
ASHA-BB | Perrone, et al. (2019)| random | yes | yes | yes | dunno
AHSA-CTS | Salinas, et al. (2021)| random | yes | yes | yes | dunno
RUSH | Zappella, et al. (2021)| random | yes | yes | yes | dunno
CBO | Gardner, et al. (2014) | ?? | yes | no | no | dunno
MOASHA | Schmucker, et al. (2021) | random | yes | yes | no | dunno

The searchers fall into three broad categories, **non-random**, **random** or **model-based**. The random searchers sample candidate hyperparameter configurations uniformly at random, while the model-based searchers sample them non-uniformly at random, according to a model (e.g., Gaussian process, density ration estimator, etc.).

BLAH on checkpointing.

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

* [Why should I use Syne Tune, and not Ray Tune, Optuna, ...?](docs/faq.md#why-syne-tune)
* [What are the different installations options supported?](docs/faq.md#installations)
* [How can I run on AWS and SageMaker?](docs/faq.md#running-on-sagemaker)
* [What are the metrics reported by default when calling the `Reporter`?](docs/faq.md#reporter-metrics)
* [How can I utilize multiple GPUs?](docs/faq.md#multiple-gpus)
* [What is the default mode when performing optimization?](docs/faq.md#default-mode)
* [How are trials evaluated on a local machine?](docs/faq.md#trial-execution)
* [What does the output of the tuning contain?](docs/faq.md#tuning-output)
* [Where can I find the output of the tuning?](docs/faq.md#tuning-output-location)
* [How can I enable trial checkpointing?](docs/faq.md#trial-checkpointing)
* [Which schedulers make use of checkpointing?](docs/faq.md#schedulers-checkpointing)
* [Is the tuner checkpointed?](docs/faq.md#tuner-checkpointing)
* [Where can I find the output of my trials?](docs/faq.md#trial-output)
* [How can I plot the results of a tuning?](docs/faq.md#plotting-tuning)
* [How can I specify additional tuning metadata?](docs/faq.md#additional-metadata)
* [How do I append additional information to the results which are stored?](docs/faq.md#logging-additional-information) 
* [I don’t want to wait, how can I launch the tuning on a remote machine?](docs/faq.md#remote-tuning)
* [How can I run many experiments in parallel?](docs/faq.md#experiment-parallel)
* [How can I access results after tuning remotely?](docs/faq.md#results-remote-tuning)
* [How can I specify dependencies to remote launcher or when using the SageMaker backend?](docs/faq.md#dependencies-remote)
* [How can I benchmark experiments from the command line?](docs/faq.md#benchmark-cli)
* [What different schedulers do you support? What are the main differences between them?](docs/faq.md#schedulers-supported)
* [How do I define the search space?](docs/faq.md#search-space) 
* [How can I visualize the progress of my tuning experiment with Tensorboard?](docs/faq.md#tensorboard) 
* [How can I add a new scheduler?](docs/faq.md#add-scheduler)
* [How can I add a new tabular or surrogate benchmark?](docs/faq.md#add-blackbox)

Do you want to know more? Here are a number of tutorials.
* [Basics of Syne Tune](docs/tutorials/basics/README.md)
* [Multi-Fidelity Hyperparameter Optimization](docs/tutorials/multifidelity/README.md)
* [How to Contribute a New Scheduler](docs/tutorials/developer/README.md)
* [Choosing a Configuration Space](docs/search_space.md)
* [Using the Built-in Schedulers](docs/schedulers.md)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## Citing Syne Tune

If you use Syne Tune in a scientific publication, please cite the following paper:

["Syne Tune: A Library for Large Scale Hyperparameter Tuning and Reproducible Research"](https://openreview.net/forum?id=BVeGJ-THIg9&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3Dautoml.cc%2FAutoML%2F2022%2FTrack%2FMain%2FAuthors%23your-submissions)) First Conference on Automated Machine Learning 2022


```bibtex
@inproceedings{
  salinas2022syne,
  title={Syne Tune: A Library for Large Scale Hyperparameter Tuning and Reproducible Research},
  author={David Salinas and Matthias Seeger and Aaron Klein and Valerio Perrone and Martin Wistuba and Cedric Archambeau},
  booktitle={First Conference on Automated Machine Learning (Main Track)},
  year={2022},
  url={https://openreview.net/forum?id=BVeGJ-THIg9}
}
```

## License

This project is licensed under the Apache-2.0 License.

