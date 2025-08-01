.. _syne_tune_concepts:

Concepts and Terminology
========================

To get started, it's helpful to understand a few key concepts.
A **configuration** is a specific set of hyperparameter values (e.g., ``learning_rate=0.01``, ``num_layers=2``). The **configuration space** defines the range of possible values for each hyperparameter. A **trial** is a single training run that evaluates a specific configuration.
In Syne Tune, an HPO experiment is coordinated by four main components: the **Tuner**, **Backend**, **Scheduler**, and **Searcher**.

Tuner
-----
The :class:`~syne_tune.Tuner` is the main entry point for your HPO experiment. You interact with it directly to start, stop, and monitor the optimization process. The Tuner acts as a conductor, coordinating the other components to find the best hyperparameter configuration. It requests new configurations to test, sends them to the backend to be executed, and collects the results.

Backend
-------
The **Backend** is the workhorse of the experiment. It is responsible for executing the actual training code for each trial. Syne Tune provides different backends for different scenarios:

*   **Local Backend**: Runs trials as separate processes on the same machine where your script is launched. This is ideal for getting started or for using a single powerful machine with multiple GPUs.
*   **Python Backend**: Works the same as the local backend but allows you to pass a Python function that defines the training logic.
*   **Simulator Backend**: Runs experiments on pre-computed data (benchmarks). This is extremely fast and useful for testing new HPO algorithms or for academic research, as it allows for reproducible and low-cost comparisons.

Searcher and Scheduler: The Brains of the Operation
---------------------------------------------------
The **Searcher** and **Scheduler** work together to decide which hyperparameter configurations to test and for how long. This is where the "intelligence" of the HPO process lies. Although they work closely, they have distinct responsibilities.

Searcher
~~~~~~~~
The **Searcher** is responsible for **proposing new hyperparameter configurations**. Think of it as the component that explores the search space. It implements a specific search algorithm to decide which configuration to try next.

Examples of searchers include:

*   **Random Search**: Samples configurations randomly from the search space.
*   **Bayesian Optimization**: Builds a probabilistic model of the objective function and uses it to select the most promising configurations to evaluate.
*   **Grid Search**: Exhaustively tries all combinations of a predefined set of hyperparameter values.

The searcher's only job is to say: "Here is a new set of hyperparameters I think we should try."

Syne Tune distinguishes between following classes of searchers:
- **BaseSearchers**: These are the core search algorithms that generate new configurations based on the search space and past results.
- **Single-Objective Searchers**: Focus on optimizing a single objective metric (e.g., validation accuracy).
- **LastValueMultiFidelitySearcher**: This implements a multi-fidelity search strategy as described by `Salinas et al`<>__, where the last observed value for each trial is passed to the searcher. It expects a SingleObjectveSearcher as input.


Scheduler
~~~~~~~~~
The **Scheduler** manages the overall experiment and the lifecycle of each trial. It takes the configurations suggested by the Searcher and decides **how to run them and for how long**. A key feature of modern HPO is the ability to stop unpromising trials early, and this logic lives in the scheduler.

For example, a simple scheduler would run each trial to completion, whereas a more advanced method like **Asynchronous Successive Halving (ASHA)** would stops the worst-performing trials early.

**The Key Difference:**

A helpful way to understand the distinction is:

*   The **Searcher** answers the question: "**What** configuration should we try next?"
*   The **Scheduler** answers the question: "**How** should we run this trial and for how long?"

A scheduler uses a searcher to get new configurations. For instance,
you can pair an `ASHA` scheduler with a `CQR` searcher. In this setup, `RandomSearch` provides the novel configurations, and `ASHA` manages the trials, promoting the good ones and stopping the bad ones early to save time and resources. This modular design allows you to mix and match different search and scheduling strategies to create powerful and customized HPO workflows.