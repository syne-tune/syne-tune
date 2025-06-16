# Syne Tune: Large-Scale and Reproducible Hyperparameter Optimization

[![release](https://img.shields.io/github/v/release/awslabs/syne-tune)](https://pypi.org/project/syne-tune/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://pepy.tech/badge/syne-tune/month)](https://pepy.tech/project/syne-tune)
[![Documentation](https://readthedocs.org/projects/syne-tune/badge/?version=latest)](https://syne-tune.readthedocs.io)
[![Python Version](https://img.shields.io/static/v1?label=python&message=3.7%20%7C%203.8%20%7C%203.9&color=blue?style=flat-square&logo=python)](https://pypi.org/project/syne-tune/)
[![codecov.io](https://codecov.io/github/awslabs/syne-tune/branch/main/graphs/badge.svg)](https://app.codecov.io/gh/awslabs/syne-tune)

![Syne Tune](docs/source/synetune.gif)

**[Documentation](https://syne-tune.readthedocs.io/en/latest/index.html)** | **[Blackboxes](https://github.com/syne-tune/syne-tune/blob/main/syne_tune/blackbox_repository/README.md)** | **[Benchmarking](https://github.com/syne-tune/syne-tune/blob/main/benchmarking/README.md)** | **[API Reference](https://syne-tune.readthedocs.io/en/latest/_apidoc/modules.html#)** | **[PyPI](https://pypi.org/project/syne-tune)** | **[Latest Blog Post](https://aws.amazon.com/blogs/machine-learning/hyperparameter-optimization-for-fine-tuning-pre-trained-transformer-models-from-hugging-face/)** | **[Discord](https://discord.gg/vzYkjZjs)** 

Syne Tune is a library for large-scale hyperparameter optimization (HPO) with the following key features:

- State-of-the-art HPO methods for multi-fidelity optimization, multi-objective optimization, transfer learning, and population-based training.

- Tooling that lets you run [large-scale experimentation](https://github.com/syne-tune/syne-tune/blob/main/benchmarking/README.md) either locally or on SLURM clusters.

- Extensive [collection of blackboxes](https://github.com/syne-tune/syne-tune/blob/main/syne_tune/blackbox_repository/README.md) including surrogate and tabular benchmarks for efficient HPO simulation.

## Installing

To install Syne Tune from pip:

```bash
pip install 'syne-tune[extra]'
```

or to install the latest version from source: 

```bash
git clone https://github.com/awslabs/syne-tune.git
cd syne-tune
pip install -e '.[extra]'
```

See our [change log](CHANGELOG.md) to see what changed in the latest version. 

## Getting started

Syne Tune assumes some python script that given hyperparameter as input arguments trains and validates a machine learning model that
somewhat follows this pattern:

```python
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--hyperparameter1', type=float)
    parser.add_argument('--hyperparameter3', type=float)
    args, _ = parser.parse_known_args()
    # instantiate your machine learning model
    for epoch in range(args.epochs):  # training loop
        # train for some steps or epoch
        ...
        # validate your model on some hold-out validation data
```

### Step 1: Adapt your training script

First, to enable tuning of your training script, you need to report metrics so they can be communicated to Syne Tune.
For example, in the script above, we assume you're tuning two hyperparameters — `height` and `width` — to minimize a loss function.
To report the loss back to Syne Tune after each epoch, simply add `report(epoch=epoch, loss=loss)` inside your training loop:

```python
# train_height_simple.py
import logging
import time

from syne_tune import Reporter
from argparse import ArgumentParser

if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--width', type=float)
    parser.add_argument('--height', type=float)
    args, _ = parser.parse_known_args()
    report = Reporter()
    for step in range(args.epochs):
        time.sleep(0.1)
        dummy_score = 1.0 / (0.1 + args.width * step / 100) + args.height * 0.1
        # Feed the score back to Syne Tune.
        report(epoch=step + 1, mean_loss=dummy_score)
```

### Step 2: Define a launching script

Once the training script is prepared, we first define the search space and then start the tuning process.
In this example, we launch [ASHA] for a total of 30 seconds using four workers.
Each worker spawns a separate Python process to evaluate a hyperparameter configuration, meaning that four configurations are trained in parallel.

```python
# launch_height_simple.py
from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint
from syne_tune.optimizer.baselines import ASHA

# hyperparameter search space to consider
config_space = {
  'width': randint(1, 20),
  'height': randint(1, 20),
  'epochs': 100,
}

tuner = Tuner(
    trial_backend=LocalBackend(entry_point='train_height_simple.py'),
    scheduler=ASHA(
        config_space,
        metric='mean_loss',
        time_attr='epoch',
    ),
    stop_criterion=StoppingCriterion(max_wallclock_time=30), # total runtime in seconds
    n_workers=4,  # how many trials are evaluated in parallel
)
tuner.run()
```

### Step 3: Plot the results

Next, we can plot the results as follows. Replace `TUNER_NAME` with the name of the tuning job 
used earlier — this is shown at the beginning of the logs.

```python
import matplotlib.pyplot as plt
from syne_tune.experiments import load_experiment

e = load_experiment('TUNER_NAME')  # name of the tuning run which is printed at the beginning of the run
e.plot_trials_over_time(metric_to_plot='mean_loss')
plt.show()
```


## Benchmarking

Checkout this tutorial to run large-scale [benchmarking](benchmarking/nursery/) with Syne Tune.

## Blog Posts

* [Run distributed hyperparameter and neural architecture tuning jobs with Syne Tune](https://aws.amazon.com/blogs/machine-learning/run-distributed-hyperparameter-and-neural-architecture-tuning-jobs-with-syne-tune/)
* [Hyperparameter optimization for fine-tuning pre-trained transformer models from Hugging Face](https://aws.amazon.com/blogs/machine-learning/hyperparameter-optimization-for-fine-tuning-pre-trained-transformer-models-from-hugging-face/) [(notebook)](https://github.com/awslabs/syne-tune/blob/hf_blog_post/hf_blog_post/example_syne_tune_for_hf.ipynb)
* [Learn Amazon Simple Storage Service transfer configuration with Syne Tune](https://aws.amazon.com/blogs/opensource/learn-amazon-simple-storage-service-transfer-configuration-with-syne-tune/) [(code)](https://github.com/aws-samples/syne-tune-s3-transfer)

## Videos

* [Martin Wistuba: Hyperparameter Optimization for the Impatient (PyData 2023)](https://www.youtube.com/watch?v=onX6fXzp9Yk)
* [David Salinas: Syne Tune: A Library for Large-Scale Hyperparameter Tuning and Reproducible Research (AutoML Seminar)](https://youtu.be/DlM-__TTa3U?feature=shared)
  
## Contributing

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## Citing Syne Tune

If you use Syne Tune in a scientific publication, please cite the following paper:

["Syne Tune: A Library for Large Scale Hyperparameter Tuning and Reproducible Research"](https://openreview.net/forum?id=BVeGJ-THIg9&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3Dautoml.cc%2FAutoML%2F2022%2FTrack%2FMain%2FAuthors%23your-submissions)) First Conference on Automated Machine Learning, 2022.


```bibtex
@inproceedings{
  salinas2022syne,
  title={Syne Tune: A Library for Large Scale Hyperparameter Tuning and Reproducible Research},
  author={David Salinas and Matthias Seeger and Aaron Klein and Valerio Perrone and Martin Wistuba and Cedric Archambeau},
  booktitle={International Conference on Automated Machine Learning, AutoML 2022},
  year={2022},
  url={https://proceedings.mlr.press/v188/salinas22a.html}
}
```

## License

This project is licensed under the Apache-2.0 License.

