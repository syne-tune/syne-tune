Benchmarking in Syne Tune
=========================

Benchmarking refers to the comparison of a range of HPO algorithms on one or
more tuning problems, or *benchmarks*. This tutorial provides an overview of
tooling which facilitates benchmarking of HPO algorithms in Syne Tune. The same
tooling can be used to rapidly create launcher scripts for any HPO experiment,
allowing you to easily switch between local, SageMaker, and simulator backend.
The tutorial also shows how any number of experiments can be run in parallel,
in order to obtain desired results faster.

.. note::
   In order to run the code in this tutorial, you need to have
   `installed Syne Tune from source <../../getting_started.html#installation>`__.
   Also, make sure to have installed the ``blackbox-repository``
   `dependencies <../faq.html#what-are-the-different-installations-options-supported>`__.

.. note::
   Benchmarking (i.e., comparing different HPO methods) is using the Syne Tune
   experimentation framework in :mod:`syne_tune.experiments`. In this framework,
   a *benchmark* is simply just a tuning problem endowed with some defaults.
   There are other use cases of experimentation than benchmarking (see
   `here <../experimentation/README.html>`__ and
   `here <../odsc_tutorial/README.html>`__), but the term *benchmark* for tuning
   problem is used in all of them.

.. toctree::
   :name: Benchmarking in Syne Tune Sections
   :maxdepth: 1

   bm_simulator
   bm_local
   bm_sagemaker
   bm_plotting
   bm_contributing
