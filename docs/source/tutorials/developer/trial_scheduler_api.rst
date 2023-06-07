The TrialScheduler API
======================

In this section, we have a closer look at the
:class:`~syne_tune.optimizer.scheduler.TrialScheduler` API, and how a scheduler
interacts with the trial backend.

Interaction between TrialScheduler and TrialBackend
---------------------------------------------------

Syne Tune supports a multitude of automatic tuning scenarios which embrace
asynchronous job execution. The goal of automatic tuning is to find a
configuration whose evaluation results in a sufficiently small (or large, if
``mode="max"``) metric value, and to do so as fast as possible. This is done
by starting trials with promising configurations (``suggest``), and
(optionally) by stopping or pausing trials which underperform. A certain
number of such evaluation (or training) jobs can be executed in parallel, on
separate *workers* (which can be different GPUs or CPU cores on the same
instance, or different instances).

In Syne Tune, this process is split between two entities: the trial backend
and the `trial scheduler <../../schedulers.html>`__. The backend wraps the
training code to be executed for different configurations and is responsible to
start jobs, as well as stop, pause or resume them. It also collects results
reported by the training jobs and relays them to the scheduler. In Syne Tune,
pause-and-resume scheduling is done via
`checkpointing <../../faq.html#how-can-i-enable-trial-checkpointing>`__. While
code to write and load checkpoints locally must be provided by the training
script, the backend makes them available when needed. There are two basic
events which happen repeatedly during an HPO experiment, as orchestrated by the
:class:`~syne_tune.Tuner`:

* The ``Tuner`` polls the backend, which signals that one or more workers are
  available. For each free worker, it calls
  :meth:`~syne_tune.optimizer.scheduler.TrialScheduler.suggest`, asking for
  what to do next. As already seen in our
  `first example <first_example.html#first-example>`__, the scheduler will
  typically suggest a configuration for a new trial to be started. On the
  other hand, a pause-and-resume scheduler may also suggest to resume a
  trial which is currently paused (having been started, and then paused,
  in the past). Based on the scheduler response, the ``Tuner`` asks the
  backend to start a new trial, or to resume an existing one.
* The ``Tuner`` polls the backend for new results, having been reported since
  the last recent poll. For each such result,
  :meth:`~syne_tune.optimizer.scheduler.TrialScheduler.on_trial_result`
  is called. The scheduler makes a decision of what to do with the reporting
  trial. Based on this decision, the ``Tuner`` asks the backend to stop or
  pause the trial (or does nothing, in case the trial is to continue).

The processing of these events is non-blocking and full asynchronous, without
any synchronization points. Depending on the backend, there can be substantial
delays between a trial reporting a result and a stop or pause decision being
executed. During this time, the training code simply continues, it may even
report further results. Moreover, a worker may be idle between finishing an
evaluation and starting or resuming another one, due to delays in the backend
or even compute time for decisions in the scheduler. However, it will never be
idle having to wait for results from other trials.

TrialScheduler API
------------------

We now discuss additional aspects of the
:class:`~syne_tune.optimizer.scheduler.TrialScheduler` API, beyond what has
already been covered `here <first_example.html#first-example>`__:

* ``suggest`` returns a
  :class:`~syne_tune.optimizer.scheduler.TrialSuggestion` object with fields
  ``spawn_new_trial_id``, ``checkpoint_trial_id``, ``config``. Here,
  :meth:`~syne_tune.optimizer.scheduler.TrialSuggestion.start_suggestion` has
  ``spawn_new_trial_id=True`` and requires ``config``. A new trial is to be
  started with configuration ``config``. Typically, this trial starts training
  from scratch. However, some specific schedulers allow the trial to warm-start
  from a checkpoint written for a different trial (an example is
  :class:`~syne_tune.optimizer.schedulers.PopulationBasedTraining`).
  A pause-and-resume scheduler may also return
  :meth:`~syne_tune.optimizer.scheduler.TrialSuggestion.resume_suggestion`,
  where ``spawn_new_trial_id=False`` and ``checkpoint_trial_id`` is mandatory.
  In this case, a currently paused trial with ID ``checkpoint_trial_id`` is to
  be resumed. Typically, the configuration of the trial does not change, but if
  ``config`` is used, the resumed trial is assigned a new configuration.
  However, for all schedulers currently implemented in Syne Tune, a trial’s
  configuration never changes.
* The only reason for ``suggest`` to return ``None`` is if no further
  suggestion can be made. This can happen if the configuration space has been
  exhausted. As discussed
  `here <first_example.html#asynchronous-job-execution>`__, the scheduler
  cannot delay a ``suggest`` decision to a later point in time.
* The helper methods ``_preprocess_config`` and ``_postprocess_config`` are
  used when interfacing with a searcher. Namely, the configuration space
  (member ``config_space``) may contain any number of fixed attributes
  alongside the hyperparameters to be tuned (the latter have values of type
  :class:`~syne_tune.config_space.Domain`), and each hyperparameter has a
  specific ``value_type`` (mostly ``float``, ``int`` or ``str``). Searchers
  require clean configurations, containing only hyperparameters with the
  correct value types, which is ensured by ``_preprocess_config``. Also,
  ``_postprocess_config`` adds back the fixed attributes from ``config_space``,
  unless they have already been set.
* ``on_trial_add``: This method is called by ``Tuner`` once a new trial has
  been scheduled to be started. In general, a scheduler may assume that if
  ``suggest`` returns
  :meth:`~syne_tune.optimizer.scheduler.TrialSuggestion.start_suggestion`, the
  corresponding trial is going to be started, so ``on_trial_add`` is not
  mandatory.
* ``on_trial_error``: This method is called by ``Tuner`` if the backend
  reports a trial’s evaluation to have failed. A useful reaction for the
  scheduler is to not propose this configuration again, and also to remove
  pending evaluations associated with this trial.
* ``on_trial_complete``: This method is called once a trial’s evaluation is
  complete, without having been stopped early. The final reported result is
  passed here. Schedulers who ignore intermediate reports from trials, may just
  implement this method and have ``on_trial_result`` return
  ``SchedulerDecision.CONTINUE``. Multi-fidelity schedulers may ignore this
  method, since any reported result is transmitted via ``on_trial_result`` (the
  final result is transmitted twice, first via ``on_trial_result``, then via
  ``on_trial_complete``).
* ``on_trial_remove`` is called when a trial gets stopped or paused, so is not
  running anymore, but also did not finish naturally. Once more, this method
  is not mandatory.
