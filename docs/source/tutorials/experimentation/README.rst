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
   In order to use the *benchmarking* framework, you need to have
   `installed Syne Tune from source <getting_started.html#installation>`__.

.. toctree::
   :name: Rapid Experimentation with Syne Tune Sections
   :maxdepth: 1

   exp_setup
   exp_plotting

