# Blackbox repository

This folder contains utilities to store and retrieve tabular evaluations into callable object.
Any stored blackbox can be combined with surrogates which allows to interpolate between configurations that were 
recorded, in addition blackbox can be used to simulate asynchronous HPO experiments with Syne Tune.

## Loading an existing blackbox

A blackbox dataset can be loaded by specifying its name and the dataset that needs to be obtained:
````python
from blackbox_repository import load
blackbox = load("nasbench201")["cifar100"]
````


The blackbox can then be called to obtain recorded evaluations:
````python
from blackbox_repository import load
blackbox = load("nasbench201")["cifar100"]
config = {k: v.sample() for k, v in blackbox.configuration_space.items()}
print(blackbox(config, fidelity={'epochs': 10}))
# {'metric_error': 0.7501,
# 'metric_runtime': 231.6001,
# 'metric_eval_runtime': 23.16001}
````

If the dataset is not found locally, it is regenerated and saved to S3 into Sagemaker bucket.

See [examples/launch_simulated_benchmark.py](../../examples/launch_simulated_benchmark.py) for examples.

## Adding a new dataset

To add a new dataset of tabular evaluations, you need to 
1) write a function able to regenerate it (see fcnet_import.py for an example)
2) add your generate in `recipes.py` in `generate_blackbox_recipe`.

## Simulating an HPO

See [examples/launch_simulated_benchmark.py](../../examples/launch_simulated_benchmark.py) for an example on how
to simulate any blackbox. You will need to specify what is the name of the objective accounting for time in order
to perform time simulation. 

