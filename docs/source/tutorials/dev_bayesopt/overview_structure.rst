Overview of Module Structure
============================

We begin with an overview of the module structure of the *Bayesian optimization
(BO)* code in Syne Tune. Feel free to directly move to the first example and come
back here for reference.

`Recall <../developer/first_example.html#searchers-and-schedulers>`__ that
Bayesian optimization is implemented in a *searcher*, which is a component of
a *scheduler* responsible for suggesting the next configuration to sample, given
data from earlier trials. While searchers using BO are located in
:mod:`syne_tune.optimizer.schedulers.searchers` and submodules, the BO code
itself is found in :mod:`syne_tune.optimizer.schedulers.searchers.bayesopt`.
`Recall <../basics/basics_bayesopt.html#what-is-bayesian-optimization>`__ that
a typical BO algorithm is configured by a *surrogate model*  and an *acquisition
function*. In Syne Tune, acquisition functions are implemented generically,
while (except for special cases) surrogate models can be grouped in two
different classes:

* Gaussian process based surrogate models: Implementations in
  :mod:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd`.
* Surrogate models based on ``scikit-learn`` like estimators: Implementations
  in :mod:`~syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn`.

The remaining code in :mod:`syne_tune.optimizer.schedulers.searchers.bayesopt`
is generic or wraps lower-level code. Submodules are as follows:

* :mod:`~syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes`:
  Collects types related to maintaining data obtained from trials. The most
  important class is
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state.TuningJobState`,
  which collects relevant data during an experiment. Note that other relevant
  classes are in :mod:`syne_tune.optimizer.schedulers.searchers.utils`, such as
  :class:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges`,
  which wraps a configuration space and maps configurations to encoded vectors
  used as inputs to a surrogate model.
* :mod:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models`:
  Contains a range of surrogate models, both for single and multi-fidelity
  tuning, along with the machinery to fit parameters of these models. In a
  nutshell, retraining of parameters and posterior computations for a surrogate
  model are defined in
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.estimator.Estimator`,
  which returns a
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.Predictor`
  to be used for posterior predictions, which in turn drive the optimization of
  an acquisition function. A model-based searcher interacts with a
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer.ModelStateTransformer`,
  which maintains the state of the experiment (a
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state.TuningJobState`
  object) and interacts with an ``Estimator``. Subclasses of ``Estimator`` and
  ``Predictor`` are mainly wrappers of underlying code in
  :mod:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd` or
  :mod:`~syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn`. Details
  will be provided shortly. This module also contains a range of acquisition
  functions, mostly in
  :mod:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc`.
* :mod:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms`:
  The Bayesian optimization logic resides here, mostly in
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm.BayesianOptimizationAlgorithm`.
  Interfaces for all relevant concepts are defined in
  :mod:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes`:

  * :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.Predictor`:
    Probabilistic predictor obtained from surrogate model, to be plugged into acquisition
    function.
  * :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.AcquisitionFunction`:
    Acquisition function, which is optimized in order to suggest the next
    configuration.
  * :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.ScoringFunction`:
    Base class of ``AcquisitionFunction`` which does not support gradient
    computations. Score functions can be used to rank a finite number of
    candidates.
  * :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.LocalOptimizer`:
    Local optimizer for minimizing the acquisition function.

* :mod:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd`:
  The Gaussian process based surrogate models, defined in
  :mod:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models`, can be
  implemented in different ways. Syne Tune currently uses the lightweight
  `autograd <https://github.com/HIPS/autograd>`__ library, and the corresponding
  implementation lies in this module.

* :mod:`~syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn`:
  Collects code required to implement surrogate models based on
  ``scikit-learn`` like estimators.

.. note::
   The most low-level code for Gaussian process based Bayesian optimization is
   contained in
   :mod:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd`, which
   is specific to `autograd <https://github.com/HIPS/autograd>`__ and L-BFGS
   optimization. Unless you want to implement a new kernel function, you
   probably do not have to extend this code. As we will see, most extensions of
   interest can be done in
   :mod:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models` (new
   surrogate model, new acquisition function), or in
   :mod:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms`
   (different BO workflow).

A Walk Through Bayesian Optimization
------------------------------------

The key primitive of BO is to suggest a next configuration to evaluate the
unknown target function at (e.g., the validation error after training a
machine learning model with a hyperparameter configuration), based on all
data gathered about this function in the past. This primitive is triggered in
the :meth:`~syne_tune.optimizer.schedulers.searchers.BaseSearcher.get_config`
method of a BO searcher. It consists of two main steps:

* Estimate surrogate model(s), given all data obtained. Often, a single surrogate
  model represents the target metric of interest, but in generalized setups
  such as multi-fidelity, constrained, or multi-objective BO, surrogate models
  may be fit to several metrics. A surrogate model provides predictive
  distributions for the metric it represents, at any configuration, which
  allows BO to explore the space of configurations not yet sampled at. For most
  built-in GP based surrogate models, estimation is done by maximizing the log
  marginal likelihood, as we see in more detail `below <gp_model.html>`__.
* Use probabilistic predictions of surrogate models to search for the best
  next configuration to sample at. This is done in
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm.BayesianOptimizationAlgorithm`,
  and is the main focus here.

:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm.BayesianOptimizationAlgorithm`
can suggest a batch of ``num_requested_candidates > 1``. If
``greedy_batch_selection == True``, this is done greedily, one configuration
at a time, yet diversity is maintained by inserting already suggested
configurations as pending into the state. If ``greedy_batch_selection == False``,
we simply return the ``num_requested_candidates`` top-scoring configurations.
For simplicity, we focus on ``num_requested_candidates == 1``, so that
a single configuration is suggested. This happens in several steps:

* First, a list of ``num_initial_candidates`` initial configurations is drawn
  at random from ``initial_candidates_generator`` of type
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.CandidateGenerator`.
* Next, these configurations are scored using ``initial_candidates_scorer`` of type
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.ScoringFunction`.
  This is a parent class of
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.AcquisitionFunction`,
  but acquisition functions support gradient computation as well. The scoring
  function typically depends on a predictor obtained from a surrogate model.
* Finally, local optimization of an acquisition function is run, using an
  instance of
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.LocalOptimizer`,
  which depends on an acquisition function and one or more predictors. Local
  optimization is initialized with the top-scoring configuration from the
  previous step. If it fails or does not result in a configuration with a
  better acquisition value, then this initial configuration is returned. The
  final local optimization can be skipped by passing an instance of
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm_components.NoOptimization`.

This workflow offers a number of opportunities for customization:

* The ``initial_candidates_generator`` by default draws configurations at random
  with replacement (checking for duplicates is expensive, and does not add
  value). This could be replaced by pseudo-random sampling with better
  coverage properties, or by Latin hypercube designs.
* The ``initial_candidate_scorer`` is often the same as the acquisition function
  in the final local optimization. Other acquisition strategies, such as
  (independent) Thompson sampling, can be implemented here.
* You may want to customize the acquisition function feeding into local
  optimization (and initial scoring), more details are provided
  `below <bo_components.html#implementing-an-acquisition-function>`__.
