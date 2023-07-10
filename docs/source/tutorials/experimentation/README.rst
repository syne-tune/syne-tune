Rapid Experimentation with Syne Tune
====================================

The main goal of automated tuning is to help the user to find and adjust the
best machine learning model as quickly as possible, given some computing
resources controlled by the user. Syne Tune contains some tooling which can
speed up this interactive process substantially. The user can launch many
experiments in parallel, slicing the complete model selection and tuning
problems into smaller parts. Comparative plots can be created from past
experimental data and easily customized to specific needs.

Syne Tune's tooling for rapid experimentation is part of the *benchmarking*
framework, which is covered in detail in
`this tutorial <../benchmarking/README.html>`__. However, as demonstrated here,
this framework is useful for experimentation beyond the comparison of different
HPO algorithm. The tutorial here is self-contained, but the reader may want
to consult the benchmarking tutorial for background information.

.. note::
   The code used in this tutorial is contained in the
   `Syne Tune sources <../../getting_started.html#installation>`__, it is not
   installed by ``pip``. You can obtain this code by installing Syne Tune from
   source, but the only code that is needed is in
   :mod:`benchmarking.examples.demo_experiment`. The final section also needs
   code from :mod:`benchmarking.nursery.odsc_tutorial`.

   Also, make sure to have installed the ``blackbox-repository``
   `dependencies <../faq.html#what-are-the-different-installations-options-supported>`__.

.. toctree::
   :name: Rapid Experimentation with Syne Tune Sections
   :maxdepth: 1

   exp_setup
   exp_plotting
   exp_packages
