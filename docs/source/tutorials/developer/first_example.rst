A First Example
===============

In this section, we start with a simple example and clarify some basic concepts.

If you have not done so, we recommend you have a look at `Basics of Syne Tune
<../basics/README.html>`__ in order to get familiar with basic concepts of Syne
Tune.

First Example
-------------

A simple example for a new scheduler (called ``SimpleScheduler``) is given by
the script
`examples/launch_height_standalone_scheduler.py <../../examples.html#launch-hpo-experiment-with-home-made-scheduler>`__.
All schedulers are subclasses of
:class:`~syne_tune.optimizer.scheduler.TrialScheduler`. Important methods
include:

* Constructor: Needs to be passed the configuration space. Most schedulers also
  have ``metric`` (name of metric to be optimized) and ``mode`` (whether metric
  is to be minimized or maximized; default is ``"min"``).
* ``_suggest`` (internal version of ``suggest``): Called by the
  :class:`~syne_tune.Tuner` whenever a worker is available. Returns trial to
  execute next, which in most cases will start a new configuration using
  trial ID ``trial_id`` (as
  :const:`~syne_tune.optimizer.scheduler.TrialSuggestion.start_suggestion`).
  Some schedulers may also suggest to resume a paused trial (as
  :const:`~syne_tune.optimizer.scheduler.TrialSuggestion.resume_suggestion`).
  Our ``SimpleScheduler`` simply draws a new configuration at random from the
  configuration space.
* ``on_trial_result``: Called by the :class:`~syne_tune.Tuner` whenever a new
  result reported
  by a running trial has been received. Here, ``trial`` provides information
  about the trial (most important is ``trial.trial_id``), and ``result``
  contains the arguments passed to :class:`~syne_tune.Reporter` by the
  underlying training script. All but the simplest schedulers maintain a
  state which is modified based on this information. The scheduler also
  decides what to do with this trial, returning a
  :class:`~syne_tune.optimizer.scheduler.SchedulerDecision` to the
  :class:`~syne_tune.Tuner`, which in turn relays this decision to the backend.
  Our ``SimpleScheduler`` maintains a sorted list of all metric values
  reported in ``self.sorted_results``. Whenever a trial reports a metric
  value which is worse than 4/5 of all previous reports (across all trials),
  the trial is stopped, otherwise it may continue. This is an example for a
  *multi-fidelity scheduler*, in that a trial reports results multiple times
  (for example, a
  script training a neural network may report validation errors at the end of
  each epoch). Even if your scheduler does not support a multi-fidelity setup,
  in that it does not make use of intermediate results, it should work properly
  with training scripts which report such results (e.g., after every epoch).
* ``metric_names``: Returns names of metrics which are relevant to this
  scheduler. These names appear as keys in the ``result`` dictionary passed to
  ``on_trial_result``.

There are further methods in
:class:`~syne_tune.optimizer.scheduler.TrialScheduler`, which will be discussed
in detail `below <trial_scheduler_api.html>`__. This simple scheduler is also
missing the ``points_to_evaluate`` argument, which we recommend every new
scheduler to support, and which is discussed in more detail
`here <random_search.html#fifoscheduler-and-randomsearcher>`__.

Basic Concepts
--------------

Recall from `Basics of Syne Tune <../basics/README.html>`__ that an HPO
experiment is run as interplay between a *backend* and a *scheduler*, which is
orchestrated by the :class:`~syne_tune.Tuner`. The backend starts, stops,
pauses, or resumes training jobs and relays their reports. A *trial* abstracts
the evaluation of a hyperparameter *configuration*. There is a diverse range of
schedulers which can be implemented in Syne Tune, some examples are:

* Simple “full evaluation” schedulers. These suggest configurations for new
  trials, but do not try to interact with running trials, even if the latter
  post intermediate results. A basic example is
  :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`, to be discussed
  `below <random_search.html#fifoscheduler-and-randomsearcher>`__.
* Early-stopping schedulers. These require trials to post intermediate results
  (e.g., validation errors after every epoch), and their ``on_trial_result``
  may stop underperforming trials early. An example is
  :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` with
  ``type="stopping"``.
* Pause-and-resume schedulers. These require trials to post intermediate
  results (e.g., validation errors after every epoch). Their ``on_trial_result``
  may pause trials at certain points in time, and their ``_suggest`` may decide
  to resume a paused trial instead of starting a new one. An example is
  :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` with
  ``type="promotion"``.

.. note::
   The method :meth:`~syne_tune.optimizer.scheduler.TrialScheduler.on_trial_result`
   returns a :class:`~syne_tune.optimizer.scheduler.SchedulerDecision`,
   signaling the tuner to *continue*, *stop*, or *pause* the reporting trial.
   The difference between *pause* and *stop* is important. If a trial is stopped,
   it cannot be resumed later on. In particular, its checkpoints may be removed
   (if the backend is created with ``delete_checkpoints=True``). On the other
   hand, if a trial is paused, it may be resumed in the future, and its most
   recent checkpoint is retained (more details are given
   `here <../../faq.html#checkpoints-are-filling-up-my-disk-what-can-i-do>`__).

Asynchronous Job Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~

One important constraint on any scheduler to be run in Syne Tune is that calls
to both ``suggest`` and ``on_trial_report`` have to be non-blocking: they need
to return instantaneously, i.e. must not wait for some future events to happen.
This is to ensure that in the presence of several workers (i.e., parallel
execution resources), idle time is avoided: Syne Tune is always executing
parallel jobs *asynchronously*.

Unfortunately, many HPO algorithms proposed in the literature assume a
synchronous job execution setup, often for conceptual simplicity (examples
include successive halving and Hyperband, as well as batch suggestions for
Bayesian optimization). In general, it just takes a little extra effort to
implement non-blocking versions of these, and Syne Tune provides ample support
code for doing so, as will be `demonstrated in detail <extend_sync_hb.html>`__.

Searchers and Schedulers
~~~~~~~~~~~~~~~~~~~~~~~~

Many HPO algorithms have a modular structure. They need to make decisions about
how to keep workers busy in order to obtain new information (``suggest``), and
they need to react to new results posted by trials (``on_trial_result``). Most
schedulers make these decisions following a general principle, such as:

* Random search: New configurations are sampled at random.
* Bayesian optimization: Surrogate models representing metrics are fit to
  result data, and they are used to make decisions (mostly ``suggest``).
  Examples include Gaussian process based BO or TPE (Tree Parzen Estimator).
* Evolutionary search: New configurations are obtained by mutating
  well-performing members of a population.

Once such internal structure is recognized, we can use it to expand the range
of methods while maintaining simple, modular implementations. In Syne Tune,
this is done by configuring generic schedulers with internal *searchers*. A
basic example is given
`below <random_search.html#fifoscheduler-and-randomsearcher>`__, more advanced
examples follow further below.

If you are familiar with `Ray Tune <https://docs.ray.io/en/latest/tune/index.html>`__,
please note a difference in terminology. In Ray Tune, searcher and scheduler
are two independent concepts, mapping to different decisions to be made by an
HPO algorithm. In Syne Tune, the HPO algorithm is represented by the scheduler,
which may have a searcher as component. We found that once model-based HPO is
embraced (e.g., Bayesian optimization), this creates strong dependencies
between suggest and stop or resume decisions, so that the supposed modularity
does not really exist.

Maybe the most important recommendation for implementing a new scheduler in
Syne Tune is this: **be lazy!**

* Can your idea be implemented as a new searcher, to be plugged into an
  existing generic scheduler? Detailed examples are given
  `here <random_search.html#fifoscheduler-and-randomsearcher>`__,
  `here <extend_async_hb.html>`__, and `here <extend_sync_hb.html>`__.
* Does your idea involve changing the stop/continue or pause/resume decisions
  in asynchronous successive halving or Hyperband? All you need to do is to
  implement a new
  :class:`~syne_tune.optimizer.schedulers.hyperband_stopping.RungSystem`.
  Examples:
  :class:`~syne_tune.optimizer.schedulers.hyperband_stopping.StoppingRungSystem`,
  :class:`~syne_tune.optimizer.schedulers.hyperband_promotion.PromotionRungSystem`,
  :class:`~syne_tune.optimizer.schedulers.hyperband_rush.RUSHStoppingRungSystem`,
  :class:`~syne_tune.optimizer.schedulers.hyperband_pasha.PASHARungSystem`,
  :class:`~syne_tune.optimizer.schedulers.hyperband_cost_promotion.CostPromotionRungSystem`.
