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

Syne Tune is a library for large-scale hyperparameter optimization (HPO) with the following key features:

- State-of-the-art HPO methods for multi-fidelity optimization, multi-objective optimization, transfer learning, and population-based training.

- Tooling that lets you run `large-scale experimentation <https://github.com/syne-tune/syne-tune/blob/main/benchmarking/README.md>`__ either locally or on SLURM clusters.

- Extensive `collection of blackboxes <https://github.com/syne-tune/syne-tune/blob/main/syne_tune/blackbox_repository/README.md>`__ including surrogate and tabular benchmarks for efficient HPO simulation.

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
   tutorials/visualization/README
   tutorials/experimentation/README
   tutorials/developer/README
   tutorials/dev_bayesopt/README
   tutorials/transfer_learning/transfer_learning
   tutorials/odsc_tutorial/README

Videos featuring Syne Tune
--------------------------

* `Martin Wistuba: Hyperparameter Optimization for the Impatient (PyData 2023) <https://www.youtube.com/watch?v=onX6fXzp9Yk>`__
* `David Salinas: Syne Tune: A Library for Large Scale Hyperparameter Tuning and Reproducible Research <https://www.youtube.com/watch?v=DlM-__TTa3U>`__

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
