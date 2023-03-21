Multi-Fidelity Hyperparameter Optimization
==========================================

This tutorial provides an overview of multi-fidelity HPO algorithms implemented
in Syne Tune. Multi-fidelity scheduling is one of the most successful recent
ideas used to speed up HPO. You will learn about the differences and relationships
between different methods, and how to choose the best approach for your own
problems.

.. note::
   This tutorial makes use of the *benchmarking* framework, you need to have
   `installed Syne Tune from source <getting_started.html#installation>`__
   to run it.

.. toctree::
   :name: Multi-Fidelity Hyperparameter Optimization Sections
   :maxdepth: 1

   mf_introduction
   mf_setup
   mf_syncsh
   mf_asha
   mf_sync_model
   mf_async_model
   mf_comparison