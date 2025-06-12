# Blackbox repository


The blackbox repository module of Syne Tune provides easy access to tabulated and surrogate benchmarks,  
which emulate the expensive evaluation of a blackbox by table lookups or predictions of a surrogate model.  
The underlying datasets are stored in an efficient format to provide fast querying times. All benchmarks are hosted  
on Hugging Face.


## Loading an existing blackbox

A blackbox dataset can be loaded by specifying its name and the dataset that needs to be obtained:

```python
from syne_tune.blackbox_repository import load_blackbox
blackbox = load_blackbox("nasbench201")["cifar100"]
```

The blackbox can then be called to obtain recorded evaluations:
```python
from syne_tune.blackbox_repository import load_blackbox
blackbox = load_blackbox("nasbench201")["cifar100"]
config = {k: v.sample() for k, v in blackbox.configuration_space.items()}
print(blackbox(config, fidelity={'hp_epoch': 10}))
# {'metric_error': 0.7501,
# 'metric_runtime': 231.6001,
# 'metric_eval_runtime': 23.16001}
```

If the dataset is not found locally, it is downloaded from the [Syne Tune HuggingFace repo](https://huggingface.co/synetune).


Some blackbox, for example PD1, do not include evaluation for all configurations in the search space. 
To use these benchmarks, we can build a surrogate model on the provided observations such that we can predict the target metrics for each configuration in the search space:
```python
from syne_tune.blackbox_repository import load_blackbox
from syne_tune.blackbox_repository.blackbox_surrogate import add_surrogate
blackbox = load_blackbox("pd1")["imagenet_resnet_batch_size_512"]
surrogate_blackbox = add_surrogate(blackbox)
config = {k: v.sample() for k, v in surrogate_blackbox.configuration_space.items()}
print(surrogate_blackbox(config, fidelity={'global_step': 10}))
```

## Simulating an HPO run

We can simulate an HPO run using the `BlackboxRepositoryBackend`, which internally uses a queuing system to simulate
asynchronous evaluation of trials:

```python
from syne_tune.blackbox_repository import load_blackbox, BlackboxRepositoryBackend
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.schedulers.asha import AsynchronousSuccessiveHalving
from syne_tune import StoppingCriterion, Tuner


n_workers = 4
blackbox_name, dataset, metric = "nasbench201", "cifar100", "metric_valid_error"
elapsed_time_attr = "metric_elapsed_time"
blackbox = load_blackbox(blackbox_name)[dataset]
trial_backend = BlackboxRepositoryBackend(
    blackbox_name=blackbox_name,
    dataset=dataset,
    elapsed_time_attr=elapsed_time_attr,
)
scheduler = AsynchronousSuccessiveHalving(
    config_space=blackbox.configuration_space,
    time_attr=blackbox.fidelity_name(),
    metric=metric,
    random_seed=31415927,
)

stop_criterion = StoppingCriterion(max_wallclock_time=7200)

# It is important to set ``sleep_time`` to 0 here (mandatory for simulator backend)
tuner = Tuner(
    trial_backend=trial_backend,
    scheduler=scheduler,
    stop_criterion=stop_criterion,
    n_workers=n_workers,
    sleep_time=0,
    # This callback is required in order to make things work with the
    # simulator callback. It makes sure that results are stored with
    # simulated time (rather than real time), and that the time_keeper
    # is advanced properly whenever the tuner loop sleeps
    callbacks=[SimulatorCallback()],
)
tuner.run()
```


## Add your own blackbox


We assume that your data includes:

- a search space  
- a list of hyperparameter configurations sampled from the search space + their performance metrics  
- optionally: a set of evaluations for multiple fidelity levels

To add your own blackbox, follow these steps:

1. Implement a new class that is derived from `BlackboxRecipe`. Crucially, this class needs to implement the function `_generate_on_disk()`, which loads your original data and stores it in our format.  
   For that, you can call the `serialize()` function, which expects a dictionary where the key specifies the task name and the value is a `BlackboxTabular` object.

2. Next, add your new class to `recipes.py` and run the following command to generate your new blackbox. To avoid your upload to the Hugging Face repo from failing, set `upload_on_hub=False`.

```bash
   python generate_and_upload_all_blackboxes.py
```

3. Lastly, you can test your blackbox by running the following script:

```bash
   python repository.py
```


## Supported blackboxes

Currently we support the following blackbox from the literature:

- FCNET by [Klein et al.](https://arxiv.org/abs/1905.04970)
- NASBench201 by [Dong et al.](https://arxiv.org/abs/2001.00326)
- ICML2020 by [Salinas et al.](https://proceedings.mlr.press/v119/salinas20a)
- LCBench by [Zimmer et al.](https://arxiv.org/abs/2006.13799)
- PD1 by [Wang et al.](https://www.jmlr.org/papers/v25/23-0269.html)
- YAHPO by [Pfisterer et al.](https://proceedings.mlr.press/v188/pfisterer22a.html)
- HPO-B by [Arango et al.](https://arxiv.org/abs/2106.06257)
- TabRepo by [Salinas et al.](https://arxiv.org/abs/2311.02971)