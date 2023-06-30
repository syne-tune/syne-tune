Implementing a Surrogate Model
==============================

In Bayesian optimization (BO), a surrogate model represents the data observed
from a target metric so far, and its probabilistic predictions at new
configurations (typically involving both predictive mean and variance) guides
the search for a most informative next acquisition. In this section, we will
show how surrogate models are implemented in Syne Tune, and give an example of
how a novel model can be added.

Recall from above that Syne Tune offers surrogate model from two broad classes:
*Gaussian process based models* and *scikit-learn estimator based models*. Both
are implemented in terms of the same abstractions
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.estimator.Estimator`,
and
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.Predictor`.
We will first walk through GP based surrogate models, then dive into an example
of how to implement a new scikit-learn estimator based model. More details
about how to extend GP based models are provided
`further below <gp_model.html>`__.


Example
-------

Before diving into details, let us look at a simple example for how to implement
a new surrogate model in Syne Tune, of the scikit-learn estimator based type. It
does not come with some of the complexities of Gaussian process based surrogate
models, to be discussed below:

* Fantasizing is not supported
* MCMC (or ensemble predictions) is not supported
* Gradient-based optimization of an acquisition function is not supported, in
  that Bayesian optimization is scoring a finite number of candidates drawn
  at random, selecting the best

The full example code is given
`here <../../examples.html#bayesian-optimization-with-scikit-learn-based-surrogate-model>`__.
We implement subclasses of
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn.SKLearnPredictor`
and
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn.SKLearnEstimator`.
These are wrapped by
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.sklearn_model.SKLearnPredictorWrapper`
and
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.sklearn_model.SKLearnEstimatorWrapper`.

.. literalinclude:: ../../../../examples/launch_sklearn_surrogate_bo.py
   :caption: examples/launch_sklearn_surrogate_bo.py
   :start-at: from syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn import (
   :end-before: if __name__ == "__main__":

* The ``BayesianRidgeEstimator`` is wrapping the scikit-learn estimator
  ``sklearn.linear_model.BayesianRidge``, which implements a form of
  Bayesian regression estimation. While this method has hyperparameters, they
  are automatically set in ``fit``, so we do not need to make them explicit.
  The result of ``fit`` is a ``BayesianRidgePredictor`` instance which wraps
  a copy of the fitted scikit-learn estimator.
* In ``BayesianRidgePredictor``, the ``predict`` methods calls the equivalent
  of the scikit-learn estimator with ``return_std=True``, so that both
  predictive means and stddevs are returned.

The remaining launcher script is much the same as other examples, except that
:class:`~syne_tune.optimizer.schedulers.FIFOScheduler` is used with a
particular searcher:

.. literalinclude:: ../../../../examples/launch_sklearn_surrogate_bo.py
   :caption: examples/launch_sklearn_surrogate_bo.py
   :start-at: searcher = SKLearnSurrogateSearcher(
   :end-before: scheduler = FIFOScheduler(

* :class:`~syne_tune.optimizer.schedulers.searchers.sklearn.SKLearnSurrogateSearcher`
  needs a :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn.SKLearnEstimator`
  object as ``estimator``, as well as the choice of acquisition function as
  ``scoring_class``.


The Predictor Class
-------------------

Scikit-learn based estimators are typically rather simple and based on
deterministic machine learning methods. Bayesian optimization is usually run with
*Bayesian models*, where proper quantification of uncertainty is center-stage,
and supporting these is a little more difficult.

In Bayesian statistics, (surrogate) models are conditioned on data in order to
obtain a posterior distribution, represented by a *posterior state*. Given this
state, probabilistic predictions can be done at arbitrary input points. This is
done by objects of type
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.Predictor`,
whose methods deal with predictions on new configurations.

.. note::
   Before moving on, it is important to understand the difference between
   *conditioning* a probabilistic model on data in order to obtain a
   posterior distribution, with which probabilistic predictions (i.e., mean
   and variance) can be computed at input points, and *learning* (or *fitting*)
   the (hyper)parameters of the model. For a Bayesian surrogate model, the
   latter involves Markov Chain Monte Carlo or marginal likelihood optimization,
   which requires conditioning on data several times. For non-Bayesian models,
   parameters are often fit by cross-validation.

At this point, there are a number of relevant concepts:

* A model can be "fitted" by Markov Chain Monte Carlo (MCMC), in which case its
  predictive distribution is an ensemble. This is why prediction methods
  returns lists. In the default case (single model, no MCMC), these lists are
  of size one.
* A model may support *fantasizing* in order to properly deal with pending
  configurations in the current state (see also ``register_pending`` in the
  discussion
  `here <../developer/random_search.html#fifoscheduler-and-randomsearcher>`__).
  At least in the Gaussian process surrogate model case, fantasizing is done
  by drawing ``nf`` samples of target values for the pending configurations,
  then average predictions over this sample. The Gaussian predictive
  distributions in this average share the same variance, but have different
  means. A surrogate model which does not support fantasizing, can ignore this
  extra complexity.

Take the example of a basic Gaussian process surrogate model, which is behind
:class:`~syne_tune.optimizer.baselines.BayesianOptimization`. The predictor is
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model.GaussProcPredictor`.
This class can serve models fit by marginal likelihood optimization (empirical
Bayes) or MCMC, but let us focus on the former. Predictions in this model are
based on a *posterior state*, which maintains a representation of the Gaussian
posterior distribution needed for probabilistic predictions. Say we would like
do a prediction at some configuration :math:`\mathbf{c}`. First, this
configuration is mapped to an (encoded) input vector :math:`\mathbf{x}`. Next,
predictive distributions are computed, using the posterior state:

.. math::
   P(y | \mathbf{x}) = \left[ \mathcal{N}(y | \mu_j(\mathbf{x}),
   \sigma^2(\mathbf{x})) \right],\quad j=1,\dots, \mathrm{nf}.

Here, ``nf`` denotes the number of fantasy samples (``nf=1`` if fantasizing is not
supported). This is served by methods of
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.Predictor`:

* ``hp_ranges_for_prediction``: Returns instance of
  :class:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges`
  which is used to map a configuration :math:`\mathbf{c}` to an encoded
  vector :math:`\mathbf{x}`.
* ``predict``: Given a matrix :math:`\mathbf{X}` of input vectors (these are
  the rows :math:`\mathbf{x}_i`), return a list of dictionaries. In our
  non-MCMC example, this list has length 1. The dictionary contains
  statistics of the predictive distribution. In our example, this would be
  predictive means (key "mean") and predictive standard deviations (key
  "std"). More precisely, the entry for "mean" would be a matrix
  :math:`[\mu_j(\mathbf{x}_i)]_{i,j}` of shape ``(n, nf)``, where ``n`` is
  the number of input vectors, and the entry for "std" would be a vector
  :math:`[\sigma(\mathbf{x}_i)]_i` of shape ``(n,)``. If the surrogate
  model does not support fantasizing, the entry for "mean" is also a
  vector of shape ``(n,)``.
* ``predict_candidates``: Version of ``predict``, where the input is a list of
  configurations :math:`[\mathbf{c}_j]`, which are first mapped to rows of
  the matrix :math:`\mathbf{X}` by using ``hp_ranges_for_prediction``.
* ``keys_predict``: Keys of dictionaries returned by ``predict``. If a
  surrogate model is to be used with a standard acquisition function, such
  as expected improvement, it needs to return at least means ("mean") and
  standard deviations ("std"). However, in other contexts, a surrogate
  model may be deterministic, in which case only means ("mean") are
  returned. This method allows an acquisition function to check whether
  it can work with surrogate models passed to it.
* ``backward_gradient``: This method is needed in order to support local
  gradient-based optimization of an acquisition function, as discussed
  `here <overview_structure.html#a-walk-through-bayesian-optimization>`__.
  It is detailed below.
* ``current_best``: A number of acquisition functions depend on the
  *incumbent*, which is a smooth approximation to the best target value
  observed so far. Typically, this is implemented as
  :math:`\mathrm{min}(\mu_j(\mathbf{x}_i))` over all inputs
  :math:`\mathbf{x}_i` already sampled for previous trials. As with
  ``predict``, this returns a list of vectors of shape ``(nf,)``,
  catering for fantasizing. If fantasizing is not supported, this is
  a list of scalars, and the list size is 1 for non-MCMC.

.. note::
   In fact,
   :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model.GaussProcPredictor`
   inherits from
   :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_base.BasePredictor`,
   which extends the base interface by some helper code to implement the
   ``current_best`` method.

Supporting Local Gradient-based Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As discussed above, BO in Syne Tune supports local gradient-based optimization
of an acquisition function. This needs to be supported by an implementation of
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.Predictor`,
in terms of the ``backward_gradient`` method.

In the most basic case, an acquisition function :math:`\alpha(\mathbf{x})` has
the following structure:

.. math::
   \alpha(\mathbf{x}) = \alpha(\mu(\mathbf{x}), \sigma(\mathbf{x})).

We ignore fantasizing here, otherwise :math:`\mu(\mathbf{x})` becomes a
vector. For gradient-based optimization, we need derivatives

.. math::
   \frac{\partial\alpha}{\partial\mathbf{x}} =
   \frac{\partial\alpha}{\partial\mu} \frac{\partial\mu}{\partial\mathbf{x}} +
   \frac{\partial\alpha}{\partial\sigma} \frac{\partial\sigma}{\partial\mathbf{x}}.

The ``backward_gradient`` method takes arguments :math:`\mathbf{x}` (``input``) and
a dictionary mapping "mean" to :math:`\partial\alpha/\partial\mu` at
:math:`\mu = \mu(\mathbf{x})`, "std" to :math:`\partial\alpha/\partial\sigma`
at :math:`\sigma = \sigma(\mathbf{x})` (``head_gradients``), and returns the
gradient :math:`\partial\alpha/\partial\mathbf{x}`.

Readers familiar with deep learning frameworks like ``PyTorch`` may wonder why
we don't just combine surrogate model and acquisition function into forming
:math:`\alpha(\mathbf{x})`, and compute its gradient by reverse mode
differentiation. However, this would strongly couple the two concepts, in that
they would have to be implemented in the same auto-differentiation system.
Instead, ``backward_gradient`` decouples the gradient computation into head
gradients for the acquisition function, which (as we will see) can be
implemented in native ``NumPy``, and ``backward_gradient`` for the surrogate
model itself. For Syne Tune's Gaussian process surrogate models, the latter
is implemented using `autograd <https://github.com/HIPS/autograd>`__. If the
``predict`` method is implemented using this framework, gradients are
obtained automatically as usual.

ModelStateTransformer and Estimator
-----------------------------------

An instance of
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.Predictor`
represents the posterior distribution of a model conditioned on observed data.
Where does this conditioning take place? Note that while machine learning
APIs like ``scikit-learn`` couple fitting and prediction in a single API, these
two are decoupled in Syne Tune by design:

* :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.estimator.Estimator`:
  The most important method is
  :meth:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.estimator.Estimator.fit_from_state`.
  It computes the *posterior state* by conditioning on observed data, which are
  sufficient statistics required for probabilistic predictions. Moreover, if
  ``update_params=True``, this final conditioning is preceded by fitting the
  (hyper)parameters of the model (this is more expensive, and if
  ``update_params=False``, the current parameters are used without updating
  them).
* :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.Predictor`:
  Wraps the posterior state computed by the ``Estimator``, allows for predictions.

The fitting of surrogate models underlying a Bayesian optimization experiment
happens in
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer.ModelStateTransformer`,
which interfaces between a model-based searcher and the surrogate model. The
``ModelStateTransformer`` maintains the state of the experiment, where all data
about observations and pending configurations are collected. Its
:meth:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer.ModelStateTransformer.fit`
method triggers fitting the surrogate models to the current data (this step can
be skipped for computational savings) and computing their posterior states.

``ModelStateTransformer`` hands down these tasks to an object of type
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.estimator.Estimator`,
which is specific to the surrogate model being used. For our Gaussian process
example, this would be
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model.GaussProcEmpiricalBayesEstimator`.
Here, parameters of the Gaussian process models (such as parameters of the
covariance function) are fitted by marginal likelihood maximization, and the
GP posterior state is computed.

.. note::
   To summarize, if your surrogate model needs to be fit to data, you need to
   implement a subclass of
   :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.estimator.Estimator`,
   whose ``fit_from_state`` method takes in data in form of a
   :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state.TuningJobState`
   and returns a
   :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.Predictor`.
   You can use
   :func:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.estimator.transform_state_to_data`
   in order to convert the ``TuningJobState`` object into the usual pair of
   feature matrix ``features`` and target vector ``targets``, along with
   normalization of targets.
