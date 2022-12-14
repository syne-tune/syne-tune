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

This package provides state-of-the-art distributed hyperparameter optimizers
(HPO) with the following key features:

* Wide coverage (>20) of different HPO methods for asynchronous optimization
  with multiple workers, including:

  * Advanced multi-fidelity methods supporting model-based decisions (including
    ASHA, BOHB, MOBSTER, DEHB, Hyper-Tune)
  * Transfer-learning optimizers that achieve better and better performance when
    used repeatedly
  * Multi-objective optimizers that can tune multiple objectives simultaneously
    (such as accuracy and latency)

* You can run HPO in different environments (locally, AWS, simulation) by
  changing one line of code. You can run many experiments in parallel on AWS
* Out-of-the-box tabulated benchmarks are available for several domains with very
  efficient simulations, which allows you to obtain results in mere seconds,
  while preserving the real dynamics of asynchronous or synchronous HPO with any
  number of workers


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

   search_space
   schedulers
   tutorials/basics/README
   tutorials/multifidelity/README
   tutorials/benchmarking/README
   tutorials/developer/README

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
