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
  * Multi-fidelity methods supporting model-based decisions (BOHB and MOBSTER);
  * Hyperparameter transfer learning to speed up (repeated) tuning jobs;
  * Multi-objective optimizers that can tune multiple objectives simultaneously (such as accuracy and latency).

* HPO can be run in different environments (locally, AWS, simulation) by changing just one line of code.
* Out-of-the-box tabulated benchmarks that allows you simulate results in seconds while preserving the real dynamics of asynchronous or synchronous HPO with any number of workers.

What's New?
-----------

* New scheduler: :class:`~syne_tune.optimizer.baselines.DyHPO`.
  This is a recent multi-fidelity method, which can be seen as alternative to
  `ASHA <tutorials/multifidelity/mf_sync_model.html>`_,
  `MOBSTER <tutorials/multifidelity/mf_async_model.html#asynchronous-mobster>`_
  or `HyperTune <tutorials/multifidelity/mf_async_model.html#hyper-tune>`_.
  Different to these, decisions on whether to promote paused trials are done
  based on the surrogate model. Our implementation differs from the published
  work by using a Gaussian process surrogate model, and by a promotion rule which
  is a hybrid between DyHPO and ASHA.
* New tutorial: `How to Contribute a New Scheduler <tutorials/developer/README.html>`_.
  Learn how to implement your own scheduler, wrap external code, or modify
  one of the existing templates in order to get your job done.
* New tutorial: `Benchmarking in Syne Tune <tutorials/benchmarking/README.html>`_.
  You'd like to run many experiments in parallel, or launch training jobs on
  different instances, all by modifying some simple scripts to your needs? Then
  our benchmarking mechanism is for you.
* You can now
  `do paired comparisons and manage seed choices <tutorials/benchmarking/bm_local.html#random-seeds-and-paired-comparisons>`_
  in order to control randomness in your comparisons.
* The `YAHPO benchmarking <tutorials/benchmarking/bm_simulator.html#the-yahpo-family>`_
  suite is integrated in our blackbox repository
* New benchmark: Transformer on WikiText-2
  (:func:`~benchmarking.commons.benchmark_definitions.transformer_wikitext2_benchmark`)


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
