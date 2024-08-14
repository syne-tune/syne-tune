Model-based Synchronous Hyperband
=================================

All methods considered so far have been extensions of random search by clever
multi-fidelity scheduling. In this section, we consider combinations of
Bayesian optimization with multi-fidelity scheduling, where configurations are
chosen based on performance of previously chosen ones, rather than being
sampled at random.

`Basics of Syne Tune: Bayesian Optimization <../basics/basics_bayesopt.html>`__
provides an introduction to Bayesian optimization in Syne Tune.

Synchronous BOHB
----------------

The first model-based method we consider is
`BOHB <https://arxiv.org/abs/1807.01774>`__, which uses the
`TPE <https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html>`__
formulation of Bayesian optimization. In the latter, an approximation to the
expected improvement (EI) acquisition function is interpreted via a ratio of
two densities. BOHB uses kernel density estimators rather than tree Parzen
estimators (as in TPE) to model the two densities.

BOHB uses the same scheduling mechanism (i.e., rung levels, promotion
decisions) than synchronous Hyperband (or SH), but it uses a model fit to past
data for suggesting the configuration of every new trial.
`Recall <mf_syncsh.html#early-stopping-hyperparameter-configurations>`__ that
validation error after :math:`r` epochs is denoted by :math:`f(\mathbf{x}, r)`,
where :math:`\mathbf{x}` is the configuration. BOHB fits KDEs separately to the
data obtained at each rung level. When a new configuration is to be suggested,
it first determines the largest rung level :math:`r_{acq}` supported by enough
data for the two densities to be properly fit. It then makes a TPE decision at
this resource level. Our `launcher script <mf_setup.html#the-launcher-script>`__
runs synchronous BOHB if ``method="BOHB"``.

API docs:

* Baseline: :class:`~syne_tune.optimizer.baselines.SyncBOHB`
* Additional arguments: :class:`~syne_tune.optimizer.schedulers.synchronous.SynchronousGeometricHyperbandScheduler`

While BOHB is often more efficient than SYNCHB, it is held back by synchronous
decision-making. Note that BOHB does not model the random function
:math:`f(\mathbf{x}, r)` directly, which makes it hard to properly react to
*pending evaluations*, i.e. trials which have been started but did not
return metric values yet. BOHB ignores pending evaluations if present, which
could lead to redundant decisions being made if the number of workers (i.e.,
parallelization factor) is large.

Synchronous MOBSTER
-------------------

Another model-based variant is synchronous
`MOBSTER <https://openreview.net/forum?id=a2rFihIU7i>`__. We will provide more
details on MOBSTER below, when discussing model-based asynchronous methods.

Our `launcher script <mf_setup.html#the-launcher-script>`__ runs synchronous
MOBSTER if ``method="SYNCMOBSTER"``. Note that the default surrogate model for
``SyncMOBSTER`` is ``gp_independent``, where the data at each rung level
is represented by an independent Gaussian process (more details are given
`here <mf_async_model.html#independent-processes-at-each-rung-level>`__).
`It turns out <mf_comparison.html>`__ that ``SyncMOBSTER`` outperforms
``SyncBOHB`` substantially on the benchmark chosen here.

API docs:

* Baseline: :class:`~syne_tune.optimizer.baselines.SyncMOBSTER`
* Additional arguments: :class:`~syne_tune.optimizer.schedulers.synchronous.SynchronousGeometricHyperbandScheduler`

When running these experiments with the simulator backend, we note that
suddenly it takes quite some time for an experiment to be finished. Still many
times faster than real time, we now need many minutes instead of seconds. This
is a reminder that *model-based decision-making can take time*. In GP-based
Bayesian optimization, hyperparameters of a Gaussian process model are fit for
every decision, and acquisition functions are being optimized over many
candidates. On the real time scale (the x axis in our result plots), this time
is often well spent. After all, ``SyncMOBSTER`` outperforms ``SyncBOHB``
significantly. But since decision-making computations cannot be tabulated, they
slow down the simulations.

As a consequence, we should be careful with result plots showing performance
with respect to number of training evaluations, as these hide both the time
required to make decisions, as well as potential inefficiencies in scheduling
jobs in parallel. HPO methods should always be compared with real experiment
time on the x axis, and the any-time performance of methods should be
visualized by plotting curves, not just quoting “final values”. Examples are
provided `here <mf_comparison.html>`__.

.. note::
   Syne Tune allows to launch experiments remotely and in parallel in order
   to still obtain results rapidly, as is detailed
   `here <../benchmarking/README.html>`__.

Differential Evolution Hyperband
--------------------------------

Another recent model-based extension of synchronous Hyperband is
`Differential Evolution Hyperband (DEHB) <https://arxiv.org/abs/2105.09821>`__.
DEHB is typically run with multiple brackets. A main difference to Hyperband
is that configurations promoted from a rung to the next are also modified by
an evolutionary rule, involving mutation, cross-over and selection. Since
configurations are not just sampled once, but potentially modified at every
rung, the hope is to find well-performing configurations faster. Our
`launcher script <mf_setup.html#the-launcher-script>`__ runs DEHB if
``method="DEHB"``.

API docs:

* Baseline: :class:`~syne_tune.optimizer.baselines.DEHB`
* Additional arguments: :class:`~syne_tune.optimizer.schedulers.synchronous.GeometricDifferentialEvolutionHyperbandScheduler`

The main feature of DEHB over synchronous Hyperband is that configurations can
be modified at every rung. However, this feature also has a drawback. Namely,
DEHB cannot make effective use of checkpointing. If a trial is resumed with a
different configuration, starting from its last recent checkpoint is not
admissable. However, our implementation is careful to make use of
checkpointing in the very first bracket of DEHB, which is equivalent to a
normal run of synchronous SH.
