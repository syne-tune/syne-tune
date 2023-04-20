Syne Tune: Large-Scale and Reproducible Hyperparameter Optimization
===================================================================

.. image:: https://img.shields.io/github/v/release/awslabs/syne-tune
   :target: https://pypi.org/project/syne-tune
   :alt: Latest Release
.. image:: https://img.shields.io/badge/3.7%20%7C%203.8%20%7C%203.9-brightgreen.svg
   :target: https://pypi.org/project/syne-tune
   :alt: Python Versions
.. image:: https://img.shields.io/github/license/awslabs/syne-tune
   :target: https://github.com/awslabs/syne-tune/blob/main/LICENSE
   :alt: License
.. image:: https://img.shields.io/pypi/dm/syne-tune
   :target: https://pepy.tech/project/syne-tune
   :alt: PyPI - Downloads

.. image:: synetune.gif

This package provides state-of-the-art algorithms for hyperparameter optimization (HPO) with the following key features:

* Wide coverage (>20) of different HPO methods, including:

  * Asynchronous versions to maximize utilization and distributed versions (i.e., with multiple workers);
  * Multi-fidelity methods supporting model-based decisions (BOHB, MOBSTER, Hyper-Tune, DyHPO, BORE);
  * Hyperparameter transfer learning to speed up (repeated) tuning jobs;
  * Multi-objective optimizers that can tune multiple objectives simultaneously (such as accuracy and latency).

* HPO can be run in different environments (locally, AWS, simulation) by changing just one line of code.
* Out-of-the-box tabulated benchmarks that allows you simulate results in seconds while preserving the real dynamics of asynchronous or synchronous HPO with any number of workers.

What's New?
-----------

* Speculative early checkpoint removal for asynchronous multi-fidelity optimization.
  Retaining all checkpoints often exhausts all available disk space when training
  large models. With this feature, Syne Tune automatically removes checkpoints
  that are unlikely to be needed.
  `Details <faq.html#checkpoints-are-filling-up-my-disk-what-can-i-do>`__.
* New Multi-Objective Scheduler:
  :class:`~syne_tune.optimizer.schedulers.multiobjective.LinearScalarizedScheduler`.
  The method works by taking a multi-objective problem and turning it into a
  single-objective task by optimizing for a linear combination of all objectives.
  This wrapper works with all single-objective schedulers. 
* Support for automatic termination criterion proposed by Makarova et al.
  Instead of defining a fix number of iterations or wall-clock time limit, we
  can set a threshold on how much worse we allow the final solution to be
  compared to the global optima, such that we automatically stop the optimization
  process once we find a solution that meets this criteria.
* You can now customize writing out results during an experiment, as shown in
  `examples/launch_height_extra_results.py <examples.html#customize-results-written-during-an-experiment>`__.
* You can now warp inputs and apply a
  `Box-Cox transform to your targets <tutorials/basics/basics_bayesopt.html#box-cox-transformation-of-target-values>`__
  in Bayesian or MOBSTER, free parameters are adjusted automatically.
* New tutorial:
  `Using Syne Tune for Transfer Learning <tutorials/transfer_learning/transfer_learning.html>`__.
  Transfer learning allows us to speed up our current optimisation by learning
  from related optimisation runs. Syne Tune provides a number of transfer HPO
  methods and makes it easy to implement new ones. Thanks to
  `Sigrid <https://github.com/sighellan>`__ for this contribution.
* New scheduler: :class:`~syne_tune.optimizer.baselines.DyHPO`.
  This is a recent multi-fidelity method, which can be seen as alternative to
  `ASHA <tutorials/multifidelity/mf_sync_model.html>`__,
  `MOBSTER <tutorials/multifidelity/mf_async_model.html#asynchronous-mobster>`__
  or `HyperTune <tutorials/multifidelity/mf_async_model.html#hyper-tune>`__.
  Different to these, decisions on whether to promote paused trials are done
  based on the surrogate model. Our implementation differs from the published
  work by using a Gaussian process surrogate model, and by a promotion rule which
  is a hybrid between DyHPO and ASHA.
* New tutorial: `Progressive ASHA <tutorials/pasha/pasha.html>`__. PASHA is a
  variant of ASHA where the maximum number of resources (e.g., maximum number
  of training epochs) is not fixed up front, but is adapted. This can lead to
  savings when training on large datasets. Thanks to
  `Ondre <https://github.com/ondrejbohdal>`__ for this contribution.


.. toctree::
   :name: Getting Started
   :caption: Getting Started
   :maxdepth: 1

   getting_started

.. toctree::
   :name: Next Steps
   :caption: Next Steps
   :maxdepth: 1

   faq_toc
   examples_toc

.. toctree::
   :name: Tutorials
   :caption: Tutorials
   :maxdepth: 1

   tutorials/basics/README
   search_space
   schedulers
   tutorials/multifidelity/README
   tutorials/benchmarking/README
   tutorials/developer/README
   tutorials/pasha/pasha
   tutorials/transfer_learning/transfer_learning

.. toctree::
   :name: API docs
   :caption: API docs
   :maxdepth: 2

   _apidoc/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
