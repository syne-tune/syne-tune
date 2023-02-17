Model-based Asynchronous Hyperband
==================================

We have seen that asynchronous decision-making tends to outperform synchronous
variants in practice, and model-based extensions of the latter can outperform
random sampling of new configurations. In this section, we discuss combinations
of Bayesian optimization with asynchronous decision-making, leading to the
currently best performing multi-fidelity methods in Syne Tune.

All examples here can either be run in stopping or promotion mode of ASHA. We
will use the promotion mode here (i.e., pause-and-resume scheduling).

Surrogate Models of Learning Curves
-----------------------------------

`Recall <mf_syncsh.html#early-stopping-hyperparameter-configurations>`_
that validation error after :math:`r` epochs is denoted by
:math:`f(\mathbf{x}, r)`, with :math:`\mathbf{x}` the configuration. Here,
:math:`r\mapsto f(\mathbf{x}, r)` is called learning curve. A learning curve
surrogate model predicts :math:`f(\mathbf{x}, r)` from observed data. A
difficult requirement in the context of multi-fidelity HPO is that observations
are much more abundant at smaller resource levels :math:`r`, while predictions
are more valuable at larger :math:`r`.

In the context of Gaussian process based
`Bayesian optimization <../basics/basics_bayesopt.html>`_, Syne Tune supports
a number of different learning curve surrogate models. The type of model is
selected upon construction of the scheduler:

.. code-block:: python

   scheduler = MOBSTER(
       config_space,
       type="promotion",
       search_options=dict(
           model="gp_multitask",
           gp_resource_kernel="exp-decay-sum",
       ),
       metric=benchmark.metric,
       mode=benchmark.mode,
       resource_attr=resource_attr,
       random_seed=random_seed,
       max_resource_attr=max_resource_attr,
   )

Here, options configuring the searcher are collected in ``search_options``. The
most important options are ``model``, selecting the type of surrogate model,
and ``gp_resource_kernel`` selecting the covariance model in the case
``model="gp_multitask"``.

Independent Processes at each Rung Level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A simple learning curve surrogate model is obtained by
``search_options["model"] = "gp_independent"``. Here, :math:`f(\mathbf{x}, r)`
at each rung level :math:`r` is represented by an independent Gaussian process
model. The models have individual constant mean functions
:math:`\mu_r(\mathbf{x}) = \mu_r` and covariance functions
:math:`k_r(\mathbf{x}, \mathbf{x}') = v_r k(\mathbf{x}, \mathbf{x}')`,
where :math:`k(\mathbf{x}, \mathbf{x}')` is a Matern-5/2 ARD kernel without
variance parameter, which is shared between the models, and the :math:`v_r > 0`
are individual variance parameters. The idea is that while validation errors at
different rung levels may be scaled and shifted, they should still exhibit
similar dependencies on the hyperparameters. The noise variance :math:`\sigma^2`
used in the Gaussian likelihood is the same across all data. However, if
``search_options["separate_noise_variances"] = True``, different noise
variances :math:`\sigma_r^2` are used for data at different rung levels.

Multi-Task Gaussian Process Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A more advanced set of learning curve surrogate models is obtained by
``search_options["model"] = "gp_multitask"`` (which is the default for
asynchronous MOBSTER). In this case, a single Gaussian process model
represents :math:`f(\mathbf{x}, r)` directly, with mean function
:math:`\mu(\mathbf{x}, r)` and covariance function
:math:`k((\mathbf{x}, r), (\mathbf{x}', r'))`. The GP model is selected by
``search_options["gp_resource_kernel"]``, currently supported options are
``"exp-decay-sum"``, ``"exp-decay-combined"``, ``"exp-decay-delta1"``,
``"freeze-thaw"``, ``"matern52"``, ``"matern52-res-warp"``,
``"cross-validation"``. The default choice is ``"exp-decay-sum"``, which is
inspired by the exponential decay model proposed
`here <https://arxiv.org/abs/1406.3896>`_. Details about these different
models are given `here <https://openreview.net/forum?id=a2rFihIU7i>`_ and in
the source code.

Decision-making is somewhat more expensive with ``"gp_multitask"`` than with
``"gp_independent"``, because the notorious cubic scaling of GP inference
applies over observations made at all rung levels. However, the extra cost is
limited by the fact that most observations by far are made at the lowest
resource level :math:`r_{min}` anyway.

Additive Gaussian Models
~~~~~~~~~~~~~~~~~~~~~~~~

Two additional models are selected by
``search_options["model"] = "gp_expdecay"`` and
``search_options["model"] = "gp_issm"``. The former is the exponential
decay model proposed `here <https://arxiv.org/abs/1406.3896>`_, the latter is
a variant thereof. These additive Gaussian models represent dependencies across
:math:`r` in a cheaper way than in ``"gp_multitask"``, and they can be fit to
all observed data, not just at rung levels. Also, joint sampling is cheap.

However, at this point, additive Gaussian models remain experimental, and they
will not be further discussed here. They can be used with MOBSTER, but not with
Hyper-Tune.

Asynchronous MOBSTER
--------------------

`MOBSTER <https://openreview.net/forum?id=a2rFihIU7i>`_ combines ASHA and
asynchronous Hyperband with GP-based Bayesian optimization. A Gaussian process
learning curve surrogate model is fit to the data at all rung levels, and
posterior predictive distributions are used in order to compute acquisition
function values and decide on which configuration to start next. We distinguish
between MOBSTER-JOINT with a GP multi-task model (``"gp_multitask"``) and
MOBSTER-INDEP with an independent GP model (``"gp_independent"``), as detailed
above. The acquisition function is expected improvement (EI) at the rung level
:math:`r_{acq}` also used by `BOHB <mf_sync_model.html#synchronous-bohb>`_.

Our `launcher script <mf_setup.html#the-launcher-script>`_ runs (asynchronous)
MOBSTER-JOINT if ``method="MOBSTER-JOINT"``. The searcher can be configured
with ``search_options``, but MOBSTER-JOINT with the ``"exp-decay-sum"``
covariance model is the default

As shown `below <mf_comparison.html>`_, MOBSTER can outperform ASHA
significantly. This is achieved by starting many less trials that stop very
early (after 1 epoch) due to poor performance. Essentially, MOBSTER rapidly
learns some important properties about the NASBench-201 problem and avoids
basic mistakes which random sampling of configurations runs into at a constant
rate. While ASHA stops such poor trials early, they still take away resources,
which MOBSTER can spend on longer evaluations of more promising configurations.
This advantage of model-based over random sampling based multi-fidelity methods
is even more pronounced when starting and stopping jobs comes with delays. Such
delays are typically present in real world distributed systems, but are absent
in our simulations.

Different to BOHB, MOBSTER takes into account *pending evaluations*, i.e.
trials which have been started but did not return metric values yet. This is
done by integrating out their metric values by Monte Carlo. Namely, we draw a
certain number of joint samples over pending targets and average the acquisition
function over these. In the context of multi-fidelity, if a trial is running, a
pending evaluation is registered for the next recent rung level it will reach.

Why is the surrogate model in MOBSTER-JOINT fit to the data at rung levels
only? After all, training scripts tend to report validation errors after each
epoch, why not use all this data? Syne Tune allows to do so (for the
``"gp_multitask"`` model), by passing ``searcher_data="all"`` when creating
the :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` (another
intermediate is ``searcher_data="rungs_and_last"``). However, while this may
lead to a more accurate model, it also becomes more expensive to fit, and does
not tend to make a difference, so the default ``searcher_data="rungs"`` is
recommended.

Finally, we can also combine ASHA with
`BOHB <mf_sync_model.html#synchronous-bohb>`_ decision-making, by choosing
``searcher="kde"`` in
:class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`. This is an
asynchronous version of BOHB.

MOBSTER-INDEP
~~~~~~~~~~~~~

Our `launcher script <mf_setup.html#the-launcher-script>`_ runs
(asynchronous) MOBSTER-INDEP if ``method="MOBSTER-INDEP"``. The independent
GPs model is selected by ``search_options["model"] = "gp_independent"``.
MOBSTER tends to perform slightly better with a joint multi-task GP model than
with an independent GPs model, justifying the Syne Tune default. In our
experience so far, changing the covariance model in MOBSTER-JOINT has only
marginal impact.

MOBSTER and Hyperband
~~~~~~~~~~~~~~~~~~~~~

Just like `ASHA can be run with multiple brackets <mf_asha.html#asynchronous-hyperband>`_,
so can MOBSTER, simply by selecting ``brackets`` when creating
:class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`. In our experience so
far, just like with ASHA, MOBSTER tends to work best with a single bracket.

Controlling MOBSTER Computations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MOBSTER often outperforms ASHA substantially. However, when applied to a problem
where many evaluations can be done, fitting the GP surrogate model to all observed
data can become slow. In fact, Gaussian process inference scales cubically in the
number of observations. The amount of computation spent by MOBSTER can be controlled:

* Setting the limit ``max_size_data_for_model``: Once the total number of
  observations is above this limit, the data is sampled down to this size. This is
  done in a way which retains all observations from trials which reached higher
  rung levels, while data from trials stopped early are more likely to be removed.
  This down sampling is redone every time the surrogate model is fit, so that
  new data (especially at higher rungs) is taken into account. Also, scheduling
  decisions about stopping, pausing, or promoting trials are always done based on
  all data.

  The default value for ``max_size_data_for_model`` is
  :const:`~syne_tune.syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.defaults.DEFAULT_MAX_SIZE_DATA_FOR_MODEL`.
  It can be changed by passing
  :code:`search_options = {"max_size_data_for_model": XYZ}` when creating the
  MOBSTER scheduler. You can switch off the limit mechanism by passing ``None``
  or a very large value. As the current default value is on the smaller end, to
  ensure fast computations, you may want to experiment with larger values as
  well.
* Parameters ``opt_skip_init_length``, ``opt_skip_period``: When fitting the GP
  surrogate model, the most expensive computation by far is refitting its own
  parameters, such as kernel parameters. The frequency of this computation can
  be regulated, as detailed
  `here <../basics/basics_bayesopt.html#speeding-up-decision-making>`_.


Hyper-Tune
----------

`Hyper-Tune <https://arxiv.org/abs/2201.06834>`_ is a model-based extension of
ASHA with some additional features compared to MOBSTER. It can be seen as
extending MOBSTER-INDEP (with the ``"gp_independent"`` surrogate model) in two
ways. First, it uses an acquisition function based on an ensemble predictive
distribution, while MOBSTER relies on the :math:`r_{acq}` heuristic from BOHB.
Second, if multiple brackets are used (Hyperband case), Hyper-Tune offers an
adaptive mechanism to sample the bracket for a new trial. Both extensions are
based on a quantification of consistency of data on different rung levels, which
is used to weight rung levels according to their reliability for making
decisions (namely, which configuration :math:`\mathbf{x}` and bracket
:math:`r_{min}` to associate with a new trial).

Our `launcher script <mf_setup.html#the-launcher-script>`_ runs Hyper-Tune
if ``method="HYPERTUNE-INDEP"``. The searcher can be configured with
``search_options``, but the independent GPs model ``"gp_independent"`` is the
default. In this example, Hyper-Tune is using a single bracket, so the
difference to MOBSTER-INDEP is due to the ensemble predictive distribution for
the acquisition function.

Syne Tune also implements Hyper-Tune with the GP multi-task surrogate models
used in MOBSTER. In result plots for this tutorial, original Hyper-Tune is
called HYPERTUNE-INDEP, while this latter variant is called HYPERTUNE-JOINT.
Our `launcher script <mf_setup.html#the-launcher-script>`_ runs this variant
if ``method="HYPERTUNE-JOINT"``.

Finally, computations of Hyper-Tune can be
`controlled in the same way as in MOBSTER <#controlling-mobster-computations>`_.

Hyper-Tune with Multiple Brackets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Just like ASHA and MOBSTER, Hyper-Tune can also be run with multiple brackets,
simply by using the ``brackets`` argument of
:class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`. If ``brackets > 1``,
Hyper-Tune samples the bracket for a new trial from an adaptive distribution
closely related to the ensemble distribution used for acquisitions. Our
`launcher script <mf_setup.html#the-launcher-script>`_ runs Hyper-Tune with 4
brackets if ``method="HYPERTUNE4-INDEP"``.

Recall that both ASHA and MOBSTER tend to work better for one than for multiple
brackets. This may well be due to the fixed, non-adaptive distribution that
brackets are sampled from. Ideally, a method would learn over time whether a
low rung level tends to be reliable in predicting the ordering at higher ones,
or whether it should rather be avoided (and :math:`r_{min}` should be
increased). This is what the adaptive mechanism in Hyper-Tune tries to do. In
our comparisons, we find that HYPERTUNE-INDEP with multiple brackets can
outperform MOBSTER-JOINT with a single bracket.

Details
~~~~~~~

In this section, we provide some details about Hyper-Tune and our
implementation. The Hyper-Tune extensions are based on a quantification of
consistency of data on different rung levels For example, assume that
:math:`r < r_{*}` are two rung levels, with sufficiently many points at
:math:`r_{*}`. If :math:`\mathcal{X}_{*}` collects trials with data at
:math:`r_{*}`, all these have also been observed at :math:`r`. Sampling
:math:`f(\mathcal{X}_{*}, r)` from the posterior distribution of the surrogate
model, we can compare the *ordering* of these predictions at :math:`r` with the
ordering of observations at :math:`r_{*}`, using a pair-wise ranking loss. A
large loss value means frequent cross-overs of learning curves between
:math:`r` and :math:`r_{*}`, and predictions at rung level :math:`r` are
unreliable when it comes to the ordering of trials :math:`\mathcal{X}_{*}` at
:math:`r_{*}`.

At any point during the algorithm, denote by :math:`r_{*}` the largest rung
level with a sufficient number of observations (our implementation requires 6
points). Assuming that :math:`r_{*} > r_{min}`, we can estimate a distribution
:math:`[\theta_r]` over rung levels :math:`\mathcal{R}_{*} =
\{r\in\mathcal{R}\, |\, r\le r_{*}\}` as follows. We draw :math:`S` independent
samples from the model at these rung levels. For each sample :math:`s`, we
compute loss values :math:`l_{r, s}` for :math:`(r, r_{*})` over all
:math:`r\in\mathcal{R}_{*}`, and determine the ``argmin`` indicator
:math:`[\text{I}_{l_{r, s} = m_s}]`, where
:math:`m_s = \text{min}(l_{r, s} | r\in\mathcal{R}_{*})`. The distribution
:math:`[\theta_r]` is obtained as normalized sum of these indicators over
:math:`s=1,\dots, S`. We also need to compute loss values :math:`l_{r_{*}, s}`,
this is done using a cross-validation approximation, see
`here <https://arxiv.org/abs/2201.06834>`_ or the code in
:mod:`syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.hypertune`
for details. In the beginning, with too little data at the second rung level,
we use :math:`\theta_{r_{min}} = 1` and 0 elsewhere.

Decisions about a new configuration are based on an acquisition function over a
predictive distribution indexed by :math:`\mathbf{x}` alone. For Hyper-Tune, an
ensemble distribution with weighting distribution :math:`[\theta_r]` is used.
Sampling from this distribution works by first sampling
:math:`r\sim [\theta_r]`, then :math:`f(\mathbf{x}) = f(\mathbf{x}, r)` from
the predictive distribution for that :math:`r`. This means that models from all
rung levels are potentially used, weighted by how reliable they predict the
ordering at the highest level :math:`r_{*}` supported by data. In our
experiments so far, this adaptive weighting can outperform the
:math:`r_{acq}` heuristic used in BOHB and MOBSTER.

Note that our implementation generalizes
`Hyper-Tune <https://arxiv.org/abs/2201.06834>`_ in that ranking losses and
:math:`[\theta_r]` are estimated once :math:`r_{*} > r_{min}` (i.e., once
:math:`r_{*}` is equal to the second rung level). In the original work, one has
to wait until :math:`r_{*} = r_{max}`, i.e. the maximum rung level is
supported by enough data. We find that for many expensive tuning problems,
early decision-making can make a large difference, so if the Hyper-Tune
extensions provide benefits, they should be used as early during the experiment
as possible. For example, in the trial plots for Hyper-Tune shown above, it
takes more than 10000 seconds for 6 trials to reach the full 200 epochs, so in
the original variant of Hyper-Tune, advanced decision-making only starts when
more than half of the experiment is already done.

If Hyper-Tune is used with more than one bracket, the :math:`[\theta_r]` is
also used in order to sample the bracket for a new trial. To this end, we need
to determine a distribution :math:`P(r)` over all rung levels which feature as
:math:`r_{min}` in a bracket. In our NASBench-201 example, if Hyper-Tune is run
with 5 brackets, the support of :math:`P(r)` would be :math:`\mathcal{S} =
\{1, 3, 9, 27, 81\}`. Also, denote the
`default distribution <mf_asha.html#asynchronous-hyperband>`_ used in ASHA
and MOBSTER by :math:`P_0(r)`. Let
:math:`r_0 = \text{min}(r_{*}, \text{max}(\mathcal{S}))`. For
:math:`r\in\mathcal{S}`, we define :math:`P(r) = M \theta_r / r` for
:math:`r\le r_0`, and :math:`P(r) = P_0(r)` for :math:`r > r_0`, where
:math:`M = \sum_{r\in\mathcal{S}, r\le r_0} P_0(r)`. In other words, we use
:math:`\theta_r / r` for rung levels supported by data, and the default
:math:`P_0(r)` elsewhere. Once more, this slightly generalizes
`Hyper-Tune <https://arxiv.org/abs/2201.06834>`_.


DyHPO
-----

`DyHPO <https://arxiv.org/abs/2202.09774>`_ is another recent model-based
multi-fidelity method. It is a promotion-based scheduler like the ones below
with ``type="promotion"``, but differs from MOBSTER and Hyper-Tune in that
promotion decisions are done based on the surrogate model, not on the
quantile-based rule of successive halving. In a nutshell:

* Rung levels are equi-spaced:
  :math:`\mathcal{R} = \{ r_{min}, r_{min} + \nu, r_{min} + 2 \nu, \dots \}`.
  If :math:`r_{min} = \nu`, this means that a trial which is promoted or
  started from scratch, always runs for :math:`\nu` resources, independent
  of its current rung level.
* Once a worker is free, we can either promote a paused trial or start a new
  one. In DyHPO, all paused trials compete with a number of new configurations
  for the next :math:`\nu` resources to be spent. The scoring criterion is a
  special version of expected improvement, so depends on the surrogate model.
* Different to MOBSTER, the surrogate model is used more frequently. Namely,
  in MOBSTER, if any trial can be promoted, the surrogate model is not
  accessed. This means that DyHPO comes with higher decision-making costs,
  which need to be controlled.
* Since scoring trials paused at the highest rung populated so far requires
  extrapolation in terms of resource :math:`r`, it cannot be used with
  ``search_options["model"] = "gp_independent"``. The other surrogate models
  are supported.

Our implementation of DyHPO differs from the published work in a number of
important points:

* `DyHPO <https://arxiv.org/abs/2202.09774>`_ uses an advanced surrogate model
  based on a neural network covariance kernel which is fitted to the current
  data. Our implementation supports DyHPO with the GP surrogate models
  detailed above, except for ``"gp_independent"``.
* Our decision rule is different from DyHPO as published, and can be seen as
  a hybrid between DyHPO and ASHA. Namely, we throw a coin :math:`\{0, 1\}`
  with probability :math:`P_1` being configurable as ``probability_sh``. If this
  gives 1, we try to promote a trial using the ASHA rule based on quantiles.
  Here, the quantile thresholds are adjusted to the linear spacing of rung
  levels. If no trial can be promoted this way, we fall back to the DyHPO rule.
  If the coin comes up 0, we use the DyHPO rule. The algorithm as published is
  obtained for :math:`P_1 = 0`. However, we find that a non-zero
  ``probability_sh`` is crucial for obtaining robust behaviour, since the
  original DyHPO rule on its own tends to start too many trials at the beginning
  before promoting any paused ones.
* Since in DyHPO, the surrogate model is used more frequently than in MOBSTER,
  it is important to control surrogate model computations, as detailed
  `above <#controlling-mobster-computations>`_. Apart from the default for
  ``max_size_data_for_model``, we also use ``opt_skip_period = 3`` as default
  for DyHPO.
