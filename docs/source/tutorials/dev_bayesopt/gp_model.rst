Combining a Gaussian Process Model from Components
==================================================

We have already seen `above <surrogate_model.html>`__ how to implement a
surrogate model from scratch. However, many Gaussian process models
proposed in the Bayesian optimization literature are combinations of more
basic underlying models. In this section, we show how such combinations
are implemented in Syne Tune.

.. note::
   When planning to implement a new Gaussian process model, you should first
   check whether the outcome is simply a Gaussian process with mean and
   covariance function arising from combinations of means and kernels of the
   components. If that is the case, it is often simpler and more efficient to
   implement a new mean and covariance function using existing code (as shown
   `above <bo_components.html#implementing-a-covariance-function-for-gp-surrogate-models>`__),
   and to use a standard GP model with these functions.

Independent Processes for Multiple Fidelities
---------------------------------------------

In this section, we will look at the example of
:mod:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent`,
providing a surrogate model for a set of functions
:math:`y(\mathbf{x}, r)`, where :math:`r\in \mathcal{R}` is an integer from a
finite set. This model is used in the context of
`multi-fidelity HPO <../multifidelity/mf_async_model.html#independent-processes-at-each-rung-level>`__.
Each :math:`y(\mathbf{x}, r)` is represented by an independent Gaussian process,
with mean function :math:`\mu_r(\mathbf{x})` and covariance function
:math:`c_r k(\mathbf{x}, \mathbf{x}')`. The covariance function :math:`k` is
shared between all the processes, but the scale parameters :math:`c_r > 0` are
different for each process. In multi-fidelity HPO, we observe more data at
smaller resource levels :math:`r`. Using the same ARD-parameterized kernel for
all processes allows to share statistical strenght between the different
levels. The code in
:mod:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent`
follows a useful pattern:

* :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.posterior_state.IndependentGPPerResourcePosteriorState`:
  Posterior state, representing the posterior distribution after conditioning
  on data. This is used (a) to compute the log marginal likelihood for fitting
  the model parameters, and (b) for predictions driving the acquisition function
  optimization.
* :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.likelihood.IndependentGPPerResourceMarginalLikelihood`:
  Wraps code to generate posterior state, and represents the negative log marginal
  likelihood function used to fit the model parameters.
* :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.gpind_model.IndependentGPPerResourceModel`:
  Wraps code for creating the likelihood object. API towards higher level code.

The code of
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.posterior_state.IndependentGPPerResourcePosteriorState`
is a simple reduction to
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_state.GaussProcPosteriorState`,
the posterior state for a basic Gaussian process. For example, here is the code
to compute the posterior state:

.. literalinclude:: ../../../../syne_tune/optimizer/schedulers/searchers/bayesopt/gpautograd/independent/posterior_state.py
   :caption: bayesopt/gpautograd/independent/posterior_state.py
   :start-at: def _compute_states(
   :end-before: def state(self, resource: int) -> GaussProcPosteriorState:

* ``mean`` and ``covariance_scale`` are dictionaries containing :math:`\mu_r`
  and :math:`c_r` respectively.
* ``features`` are extended features of the form :math:`(\mathbf{x}_i, r_i)`.
  The function ``decode_extended_features`` maps this to arrays
  :math:`[\mathbf{x}_i]` and :math:`[r_i]`.
* We compute separate posterior states for each level :math:`r\in\mathcal{R}`,
  using the data :math:`(\mathbf{x}_i, y_i)` so that :math:`r_i = r`.
* Other methods of the base class
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_state.PosteriorStateWithSampleJoint`
  are implemented accordingly, reducing computations to the states for each
  level.

The code of
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.likelihood.IndependentGPPerResourceMarginalLikelihood`
is obvious, given the base class
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.likelihood.MarginalLikelihood`.
The same holds for
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.gpind_model.IndependentGPPerResourceModel`,
given the base class
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gp_model.GaussianProcessOptimizeModel`.
One interesting feature is that the creation of the likelihood object is
delayed, because the set of rung levels :math:`\mathcal{R}` of the multi-fidelity
scheduler need to be known. The ``create_likelihood`` method is called in
:meth:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model.GaussProcEstimator.configure_scheduler`,
a callback function with the scheduler as argument.

Since our independent GP model implements the APIs of
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.likelihood.MarginalLikelihood`
and
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gp_model.GaussianProcessOptimizeModel`,
we can plug it into generic code in :mod:`syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model`,
which works as outlined
`above <surrogate_model.html#modelstatetransformer-and-estimator>`__.
In particular, the estimator
:mod:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model.GaussProcEmpiricalBayesEstimator`
accepts ``gp_model`` of type
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.gpind_model.IndependentGPPerResourceModel`,
and it creates predictors of type
:mod:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model.GaussProcPredictor`.

Overview of ``gpautograd``
--------------------------

Most of the code in
:mod:`syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd` adheres to
the same pattern (posterior state, likelihood function, model wrapper):

* Standard GP model:
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_state.GaussProcPosteriorState`,
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.likelihood.GaussianProcessMarginalLikelihood`,
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gp_regression.GaussianProcessRegression`.
  This also covers multi-task GP models for multi-fidelity, by way of extended
  configurations.
* Independent GP models for multi-fidelity (example above):
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.posterior_state.IndependentGPPerResourcePosteriorState`,
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.likelihood.IndependentGPPerResourceMarginalLikelihood`,
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.gpind_model.IndependentGPPerResourceModel`.
* Hyper-Tune independent GP models for multi-fidelity:
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.hypertune.posterior_state.HyperTuneIndependentGPPosteriorState`,
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.hypertune.likelihood.HyperTuneIndependentGPMarginalLikelihood`,
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.hypertune.gp_model.HyperTuneIndependentGPModel`.
* Hyper-Tune multi-task GP models for multi-fidelity:
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.hypertune.posterior_state.HyperTuneJointGPPosteriorState`,
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.hypertune.likelihood.HyperTuneJointGPMarginalLikelihood`,
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.hypertune.gp_model.HyperTuneJointGPModel`.
* Linear state space learning curve models:
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.posterior_state.IncrementalUpdateGPAdditivePosteriorState`,
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.likelihood.GaussAdditiveMarginalLikelihood`,
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.gpiss_model.GaussianProcessLearningCurveModel`.
  This code is still experimental.
