Extending Asynchronous Hyperband
================================

Syne Tune provides powerful generic scheduler templates for popular
methods like successive halving and Hyperband. These can be run with
synchronous or asynchronous decision-making. The most important generic
templates at the moment are:

* `FIFOScheduler <random_search.html#fifoscheduler-and-randomsearcher>`__:
  *Full evaluation* scheduler, baseclass for many others. See also
  :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`.
* `HyperbandScheduler <extend_async_hb.html#hyperbandscheduler>`__:
  Asynchronous successive halving and Hyperband. See also
  :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.
* `SynchronousHyperbandScheduler <extend_sync_hb.html#synchronous-hyperband>`__:
  Synchronous successive halving and Hyperband. See also
  :class:`~syne_tune.optimizer.schedulers.synchronous.SynchronousHyperbandScheduler`.

Chances are your idea for a new scheduler maps to one of these templates, in
which case you can save a lot of time and headache by just extending the
template, rather than re-implementing the wheel. Due to Syne Tune’s modular
design of schedulers and their components (e.g., searchers, decision rules),
you may even get more than you bargained for.

In this section, we will walk through an example of how to furnish the
asynchronous successive halving scheduler with a specific searcher.

HyperbandScheduler
------------------

Details about asynchronous successive halving and Hyperband are given in the
`Multi-fidelity HPO tutorial <../multifidelity/README.html>`__. This is a
multi-fidelity scheduler, where trials report intermediate results (e.g.,
validation error at the end of each epoch of training). We can formalize this
notion by the concept of *resource* :math:`r = 1, 2, 3, \dots` (e.g.,
:math:`r` is the number of epochs trained). A generic implementation of this
method is provided in class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.
Let us have a look at its arguments not shared with the base class
class:`~syne_tune.optimizer.schedulers.FIFOScheduler`:

* A mandatory argument is ``resource_attr``, which is the name of a field in
  the ``result`` dictionary passed to ``scheduler.on_trial_report``. This field
  contains the resource :math:`r` for which metric values have been reported.
  For example, if a trial reports validation error at the end of the 5-th epoch
  of training, ``result`` contains ``{resource_attr: 5}``.
* We already noted the arguments ``max_resource_attr`` and ``max_t`` in
  class:`~syne_tune.optimizer.schedulers.FIFOScheduler`. They are used to
  determine the maximum resource :math:`r_{max}` (e.g., the total number of
  epochs a trial is to be trained, if not stopped before). As discussed in
  detail `here <../multifidelity/mf_setup.html#the-launcher-script>`__, it is
  best practice reserving a field in the configuration space
  ``scheduler.config_space`` to contain :math:`r_{max}`. If this is done, its
  name should be passed in ``max_resource_attr``. Now, every configuration sent
  to the training script contains :math:`r_{max}`, which should not be hardcoded
  in the script. Moreover, if ``max_resource_attr`` is used, a pause-and-resume
  scheduler (e.g., :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`
  with ``type="stopping"``) can modify this field in the configuration of a trial
  which is only to be run until a certain resource less than :math:`r_{max}`.
  Nevertheless, if ``max_resource_attr`` is not used, then :math:`r_{max}` has
  to be passed explicitly via ``max_t`` (which is not needed if
  ``max_resource_attr`` is used).
* ``reduction_factor``, ``grace_period``, ``brackets`` are important parameters
  detailed in the `tutorial <../multifidelity/README.html>`__. If
  ``brackets > 1``, we run asynchronous Hyperband with this number of brackets,
  while for ``bracket == 1`` we run asynchronous successive halving (this is the
  default).
* As detailed in the
  `tutorial <../multifidelity/mf_asha.html#asynchronous-successive-halving-early-stopping-variant>`__,
  ``type`` determines whether the method uses early stopping (``type="stopping"``)
  or pause-and-resume scheduling (``type="promotion"``). Further choices of
  ``type`` activate specific algorithms such as RUSH, PASHA, or cost-sensitive
  successive halving.

Kernel Density Estimator Searcher
---------------------------------

One of the most flexible ways of extending
:class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` is to provide it with
a novel `searcher <first_example.html#searchers-and-schedulers>`__. In order to
understand how this is done, we will walk through
:class:`~syne_tune.optimizer.schedulers.searchers.kde.MultiFidelityKernelDensityEstimator`
and
:class:`~syne_tune.optimizer.schedulers.searchers.kde.KernelDensityEstimator`.
This searcher implements ``suggest`` as in
`BOHB <https://arxiv.org/abs/1807.01774>`__, as also detailed in
`this tutorial <../multifidelity/mf_sync_model.html#synchronous-bohb>`__. In a
nutshell, the searcher splits all observations into two parts (*good* and
*bad*), depending on metric values lying above or below a certain quantile, and
fits kernel density estimators to these two subsets. It then makes decisions
based on a particular ratio of these densities, which is approximating a
variant of the expected improvement acquisition function.

We begin with the base class
:class:`~syne_tune.optimizer.schedulers.searchers.kde.KernelDensityEstimator`,
which works with schedulers implementing
:class:`~syne_tune.optimizer.schedulers.scheduler_searcher.TrialSchedulerWithSearcher`
(the most important one being :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`),
but already implements most of what is needed in the multi-fidelity context.

* The code does quite some bookkeeping concerned with mapping configurations to
  feature vectors. If you want to do this from scratch for your searcher, we
  recommend to use
  :class:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges`.
  However, ``KernelDensityEstimator`` was extracted from the original BOHB
  implementation.
* Observation data is collected in ``self.X`` (feature vectors for
  configurations) and ``self.y`` (values for ``self._metric``, negated if
  ``self.mode == "max"``). In particular, the ``_update`` method simply appends
  new data to these members.
* ``get_config`` fits KDEs to the good and bad parts of ``self.X``, ``self.y``.
  It then samples ``self.num_candidates`` configurations at random, evaluates
  the TPE acquisition function for each candidate, and returns the best one.
* ``configure_scheduler`` is a callback which allows the searcher to check whether
  its scheduler is compatible, and to depend on details of this scheduler.
  In our case, we check whether the scheduler implements
  :class:`~syne_tune.optimizer.schedulers.scheduler_searcher.TrialSchedulerWithSearcher`,
  which is the minimum requirement for a searcher.

.. note::
   Any scheduler configured by a searcher should inherit from
   :class:`~syne_tune.optimizer.schedulers.scheduler_searcher.TrialSchedulerWithSearcher`,
   which mainly makes sure that
   :meth:`~syne_tune.optimizer.schedulers.searchers.BaseSearcher.configure_scheduler`
   is called before the searcher is first used. It is also strongly recommended
   to implement ``configure_scheduler`` for a new searcher, restricting usage
   to compatible schedulers.

The class
:class:`~syne_tune.optimizer.schedulers.searchers.kde.MultiFidelityKernelDensityEstimator`
inherits from ``KernelDensityEstimator``:

* On top of ``self.X`` and ``self.y``, it also maintains resource values
  :math:`r` for each datapoint in ``self.resource_levels``.
* ``get_config`` remains the same, only its subroutine ``train_kde`` for
  training the good and bad density models is modified. The idea is to fit
  these to data from a single rung level, namely the largest level at which we
  have observed at least ``self.num_min_data_points`` points.
* ``configure_scheduler`` restricts usage to schedulers implementing
  :class:`~syne_tune.optimizer.schedulers.multi_fidelity.MultiFidelitySchedulerMixin`,
  which all multi-fidelity schedulers need to inherit from (examples are
  :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` for asynchronous
  Hyperband and
  :class:`~syne_tune.optimizer.schedulers.synchronous.SynchronousHyperbandScheduler`
  for synchronous Hyperband). It also calls
  :meth:`~syne_tune.optimizer.schedulers.searchers.kde.KernelDensityEstimator.configure_scheduler`.
  Moreover, ``self.resource_attr`` is obtained from the scheduler, so does not
  have to be passed.

.. note::
   Any *multi-fidelity* scheduler configured by a searcher should inherit from both
   :class:`~syne_tune.optimizer.schedulers.scheduler_searcher.TrialSchedulerWithSearcher` and
   :class:`~syne_tune.optimizer.schedulers.multi_fidelity.MultiFidelitySchedulerMixin`.
   The latter is a basic API to be implemented by multi-fidelity schedulers, which
   is used by the ``configure_scheduler`` of searchers specialized to multi-fidelity
   HPO. Doing so makes sure any new multi-fidelity scheduler can seamlessly be
   used with any such searcher.

While being functional and simple, the
``MultiFidelityKernelDensityEstimator`` does not showcase the full range of
information exchanged between ``HyperbandScheduler`` and a searcher. In
particular:

* ``register_pending``: BOHB does not take pending evaluations into account.
* ``remove_case``, ``evaluation_failed`` are not implemented.
* ``get_state``, ``clone_from_state`` are not implemented, so schedulers with
  this searcher are not properly serialized.

For a more complete and advanced example, the reader is invited to study
:class:`~syne_tune.optimizer.schedulers.searchers.GPMultiFidelitySearcher` and
:class:`~syne_tune.optimizer.schedulers.searchers.GPFIFOSearcher`.
This searcher takes pending evaluations into account (by way of fantasizing).
Moreover, it can be configured with a Gaussian process model and an acquisition
function, which is optimized in a gradient-based manner.

Moreover, as already noted `here <first_example.html#searchers-and-schedulers>`__,
``HyperbandScheduler`` also allows to configure the decision rule for
stop/continue or pause/resume as part of ``on_trial_report``. Examples for this
are found in
:class:`~syne_tune.optimizer.schedulers.hyperband_stopping.StoppingRungSystem`,
:class:`~syne_tune.optimizer.schedulers.hyperband_promotion.PromotionRungSystem`,
:class:`~syne_tune.optimizer.schedulers.hyperband_rush.RUSHStoppingRungSystem`,
:class:`~syne_tune.optimizer.schedulers.hyperband_pasha.PASHARungSystem`,
:class:`~syne_tune.optimizer.schedulers.hyperband_cost_promotion.CostPromotionRungSystem`.
