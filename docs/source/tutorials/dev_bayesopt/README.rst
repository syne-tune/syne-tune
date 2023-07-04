How to Implement Bayesian Optimization
======================================

This tutorial can be seen as more advanced successor of our
`developer tutorial <../developer/README.html>`__. It provides an overview of
how model-based search, and in particular *Bayesian optimization*, is
implemented in Syne Tune, and how this code can be extended in order to fit your
needs. The basic developer tutorial is a prerequisite to take full advantage
of the advanced tutorial here.

We hope this information inspires you to give it a try to extend Syne Tune's
Bayesian optimization to your needs. Please do consider
`contributing your efforts to Syne Tune <https://github.com/awslabs/syne-tune/blob/main/CONTRIBUTING.md>`__.

.. note::
   In order to develop new methodology in Syne Tune, make sure to use an
   `installation from source <../../faq.html#what-are-the-different-installation-options-supported>`__.
   In particular, you need to have installed the ``dev`` dependencies.

.. toctree::
   :name: How to Implement Bayesian Optimization Sections
   :maxdepth: 1

   overview_structure
   surrogate_model
   bo_components
   gp_model
