Wrapping External Scheduler Code
================================

One of the most common instances of extending Syne Tune is wrapping external
code. While there are comprehensive open source frameworks for HPO, many
recent advanced algorithms are only available as research codes, typically
ignoring systems aspects such as distributed scheduling, or maintaining results
in an interchangeable format. Due to the modular, backend-agnostic design of
Syne Tune, external scheduler code is easily integrated, and can then be
compared "apples to apples" against a host of baselines, be it by fast simulation
on surrogate benchmarks, or distributed across several machines.

In this chapter, we will walk through an example of how to wrap Gaussian
process based Bayesian optimization from
`BoTorch <https://botorch.org/docs/introduction>`__.

BoTorchSearcher
---------------

While Syne Tune supports Gaussian process based Bayesian optimization natively
via :class:`~syne_tune.optimizer.schedulers.searchers.GPFIFOSearcher`, with
``searcher="bayesopt"`` in :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`,
you can also use `BoTorch <https://botorch.org/docs/introduction>`__ via
:class:`~syne_tune.optimizer.schedulers.searchers.botorch.BoTorchSearcher`,
with ``searcher="botorch"`` in
:class:`~syne_tune.optimizer.schedulers.FIFOScheduler`.

Before we look into the code, note that even though we wrap external HPO code,
we still need to implement some details on our side:

* We need to maintain the trials which have resulted in observations, as well
  as those which are pending (e.g., have been started, but have not yet returned
  an observation).
* We need to provide the code for suggesting initial configurations, either
  drawing from ``points_to_evaluate``, or sampling at random.
* We need to avoid duplicate suggestions if ``allow_duplicates == False``.
* BoTorch requires configurations to be encoded as vectors with values in
  :math:`[0, 1]`. We need to provide this encoding and decoding as well.

Such details are often ignored in research code (in fact, most HPO code just
implements the equivalent of
:meth:`~syne_tune.optimizer.schedulers.searchers.BaseSearcher.get_config`,
given all previous data), but has robust and easy to use solutions in Syne Tune,
as we demonstrate here. Here is
:meth:`~syne_tune.optimizer.schedulers.searchers.botorch.BoTorchSearcher._get_config`:

.. literalinclude:: ../../../../syne_tune/optimizer/schedulers/searchers/botorch/botorch_searcher.py
   :caption: syne_tune/optimizer/schedulers/searchers/botorch/botorch_searcher.py
   :start-at: def _get_config(self, trial_id: str, **kwargs)
   :end-before: def register_pending(

* First, :code:`self._next_initial_config()` is called, which returns a
  configuration from ``points_to_evaluate`` if there is still one not yet
  returned, otherwise ``None``.
* Otherwise, if there are less than :code:`self.num_minimum_observations` trials
  which have returned observation, we return a randomly sampled configuration
  (:code:`self._get_random_config()`), otherwise one suggested by BoTorch
  (:code:`self._sample_next_candidate()`).
* Here, :code:`self._get_random_config()` is implemented in the base class
  :class:`~syne_tune.optimizer.schedulers.searchers.StochasticAndFilterDuplicatesSearcher`
  and calls the same code as all other schedulers employing random suggestions
  in Syne Tune. In particular, this function allows to pass an exclusion list
  of configurations to avoid.
* The exclusion list :code:`self._excl_list` is maintained in the base class
  :class:`~syne_tune.optimizer.schedulers.searchers.StochasticAndFilterDuplicatesSearcher`.
  If ``allow_duplicates == False``, it contains all configurations suggested
  previously. Otherwise, it contains configurations of failed or pending trials,
  which we want to avoid in any case. The exclusion list is implemented as
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.common.ExclusionList`.
  Configurations are represented by hash strings which are independent of details
  such as floating point resolution.
* If ``allow_duplicates == False`` and the configuration space is finite, it can
  happen that all configurations have already been suggested, in which case
  ``get_config`` returns ``None``.
* Finally, ``_get_config`` is called in
  :meth:`~syne_tune.optimizer.schedulers.searchers.StochasticAndFilterDuplicatesSearcher.get_config`,
  where if ``allow_duplicates == False``, the new configuration is added to the
  exclusion list.
* In :meth:`~syne_tune.optimizer.schedulers.searchers.botorch.BoTorchSearcher._sample_next_candidate`,
  the usage of :code:`self._restrict_configurations` is of interest. It relates to
  the ``restrict_configurations`` argument. If this is not
  ``None``, configurations are suggested from a finite set, namely those in
  :code:`self._restrict_configurations`. If ``allows_duplicates == False``,
  entries are removed from there once suggested. For our example, we need to avoid
  doing a local optimization of the acquisition function (via :code:`optimize_acqf`)
  in this case, but use
  :meth:`~syne_tune.optimizer.schedulers.searchers.botorch.BoTorchSearcher._sample_and_pick_acq_best`
  instead. Since the latter uses :code:`self._get_random_config()`, we are all set,
  since this makes use of :code:`self._restrict_configurations` already.

Other methods are straightforward:

* We also take care of pending evaluations (i.e. trials whose observations have
  not been reported yet). In
  :meth:`~syne_tune.optimizer.schedulers.searchers.botorch.BoTorchSearcher.register_pending`,
  the trial ID is added to :code:`self.pending_trials`.
* :meth:`~syne_tune.optimizer.schedulers.searchers.botorch.BoTorchSearcher._update`
  stores the metric value from :code:`result[self._metric]`, where
  :code:`self._metric` is the name of the primary metric. Also, the trial is
  removed from :code:`self.pending_trials`, so it ceases to be pending.
* By implementing
  :meth:`~syne_tune.optimizer.schedulers.searchers.botorch.BoTorchSearcher.evaluation_failed`
  and
  :meth:`~syne_tune.optimizer.schedulers.searchers.botorch.BoTorchSearcher.cleanup_pending`,
  we make sure that failed trials do not remain pending.
* :meth:`~syne_tune.optimizer.schedulers.searchers.botorch.BoTorchSearcher.configure_scheduler`
  is a callback which allows the searcher to depend on its scheduler. In
  particular, the searcher should reject non-supported scheduler types. The base
  class implementation
  :meth:`~syne_tune.optimizer.schedulers.searchers.BaseSearcher.configure_scheduler`
  sets :code:`self._metric` and :code:`self._mode` from the corresponding attributes
  of the scheduler, so they do not have to be set at construction of the
  searcher.

Finally, all the code specific to BoTorch is located in
:meth:`~syne_tune.optimizer.schedulers.searchers.botorch.BoTorchSearcher._sample_next_candidate`
and other internal methods. Importantly, BoTorch requires configurations to be
encoded as vectors with values in :math:`[0, 1]`, which is done using the
:code:`self._hp_ranges` member, as is detailed below.

.. note::
   When implementing a new searcher, whether from scratch or wrapping external
   code, we recommend you use the base class
   :class:`~syne_tune.optimizer.schedulers.searchers.StochasticAndFilterDuplicatesSearcher`
   and implement the ``allow_duplicates`` argument. This will also give you
   proper random seed management and ``points_to_evaluate``. Instead of
   ``get_config``, you implement the internal method ``_get_config``. If you need
   to draw configurations at random, use the method ``_get_random_config`` which
   uses the built-in exclusion list, properly deals with configuration spaces
   of finite size, and uses the random generator seeded in a consistent and
   reproducible way.

   We also recommend that you implement the ``restrict_configurations`` argument,
   unless this is hard to do for your scheduler. Often, a scheduler can be made
   to score a certain number of configurations and return the best. If so, you
   use ``self._get_random_config()`` to select the configurations to score, which
   take care of ``restrict_configurations``.

HyperparameterRanges
--------------------

Most model-based HPO algorithms require configurations to be encoded as vectors
with values in :math:`[0, 1]`. If :math:`\mathbf{u} = e(\mathbf{x})` and
:math:`\mathbf{x} = d(\mathbf{u})` denote encoding and decoding map, where
:math:`\mathbf{x}\in \mathcal{X}` is a configuration and
:math:`\mathbf{u} \in [0,1]^k`, then :math:`d(e(\mathbf{x})) = \mathbf{x}` for
every configuration :math:`\mathbf{x}`, and a random sample :math:`d(\mathbf{u})`,
where the components of :math:`\mathbf{u}` are sampled uniformly at random, is
equivalent to a random sample from the configuration space, as defined by the
hyperparameter domains.

With :class:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges`,
Syne Tune provides encoding and decoding for all domains in
:mod:`syne_tune.config_space` (see `this tutorial <../search_space.html>`__ for
a summary). In fact, this API can be implemented in different ways, and the
factory function
:func:`~syne_tune.optimizer.schedulers.searchers.utils.make_hyperparameter_ranges`
can be used to create a ``HyperparameterRanges`` object from a configuration
space.

* :meth:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges.to_ndarray`
  provides the encoding map :math:`e(\mathbf{x})`, and
  :meth:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges.to_ndarray_matrix`
  encodes a list of configurations into a matrix.
* :meth:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges.from_ndarray`
  provides the decoding map :math:`d(\mathbf{u})`.
* :meth:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges.config_to_match_string`
  maps a configuration to a hash string which can be used to test for (approximate)
  equality (see ``allow_duplicates`` discussion above).

Apart from encoding and decoding, ``HyperparameterRanges`` provides further
functionalities, such as support for a resource attribute in model-based
multi-fidelity schedulers, or the ``active_config_space`` feature which is
useful to support transfer tuning (i.e., HPO in the presence of evaluation
data from earlier experiments with different configuration spaces).

.. note::
   When implementing a new searcher or wrapping external code, we recommend you
   use :class:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges`
   in order to encode and decode configurations as vectors, instead of writing
   this on your own. Doing so ensures that your searcher supports all
   hyperparameter domais offered by Syne Tune, even new ones potentially added
   in the future. If you do not like the built-in implementation of the
   ``HyperparameterRanges`` API, feel free to contribute a different one.

Managing Dependencies
---------------------

External code can come with extra dependencies. For example, ``BoTorchSearcher``
depends on ``torch``, ``botorch``, and ``gpytorch``. If you just use Syne Tune
for your own experiments, you do not have to worry about this. However, we
strongly encourage you to
`contribute back your extension <https://github.com/awslabs/syne-tune/blob/main/CONTRIBUTING.md>`__.

Since some applications of Syne Tune require restricted dependencies, such are
carefully managed. There are different
`installation options <../../faq.html#what-are-the-different-installation-options-supported>`__,
each of which coming with a ``requirements.txt`` file (see ``setup.py`` for
details).

* First, check whether any of the installation options cover the dependencies
  of your extension (possibly a union of several of them). If so, please use
  conditional imports w.r.t. these (see below)
* If the required dependencies are not covered, you can create a new
  installation option (say, ``foo``), via ``requirements-foo.txt`` and a
  modification of ``setup.py``. In this case, please also extend
  :mod:`~syne_tune.try_import` by a function ``try_import_foo_message``.

Once all required dependencies are covered by some installation option, wrap
their imports as follows:

.. code-block:: python

   try:
       from foo import bar  # My dependencies
       # ...
   except ImportError:
       print(try_import_foo_message())
