Extending the Documentation
===========================

Syne Tune comes with an extensive amount of documentation:

* User-facing APIs are commented in the code, using the reStructered text format.
  This is used to generate the *API Reference*. Please refer to the code in
  order to understand our conventions. Please make sure that links to classes,
  methods, or functions work. In the presence of ``:math:`` expression, the
  docstring should be raw: ``r""" ... """``.
* Examples in ``examples/`` are working, documented scripts showcasing
  individual features. If you contribute a new example, please also link it
  in `docs/source/examples.rst <../../examples.html>`__.
* Frequently asked questions at
  `docs/source/faq.rst <../../faq.html>`__.
* Table of all HPO algorithms in
  `docs/source/getting_started.rst <../../getting_started.html#supported-hpo-methods>`__.
  If you contribute a new HPO method, please add a row there. As explained above,
  please also extend :mod:`~syne_tune.optimizer.baselines`.
* Tutorials at ``docs/source/tutorials/``. These are short chapters, explaining
  a concept in more detail than an example. A tutorial should be self-contained
  and come with functioning code, which can be run in a reasonable amount of
  time and cost. It may contain figures created with a larger effort.

Building the Documentation
--------------------------

You can build the documentation locally as follows. Make sure to have Syne
Tune installed with ``dev`` dependencies:

.. code-block:: bash

   cd docs
   rm -rf source/_apidoc
   make clean
   make html

Then, open ``docs/build/html/index.html`` in your browser.

The documentation is also built as part of our CI system, so you can inspect it
as part of a pull request:

* Move to the list of all checks (if the PR is in good shape, you should see
  *All checks have passed*)
* Locate **docs/readthedocs.org:syne-tune** at the end of the list. Click on
  *Details*
* Click on *View docs* just below *Build took X seconds* (do not click on the
  tall *View Docs* button upper right, this leads to the latest public docs)

When extending the documentation, please verify the following:

* Check whether links work. They typically fail silently, possibly emitting
  a warning. Use proper links when referring to classes, modules, functions,
  methods, or constants, and check whether the links to the API Reference
  work.

Conventions
-----------

We use the following conventions to ensure that documentation stays
up-to-date:

* Use ``literalinclude`` for almost all code snippets. In general, the
  documentation is showing code which is part of a functional script,
  which can either be in ``examples/``, in ``benchmarking/examples/``, or
  otherwise next to the documentation files.
* Almost all code shown in the documentation is run as part of
  integration testing (``.github/workflows/integ-tests.yml``) or
  end-to-end testing (``.github/workflows/end-to-end-tests.yml``). If you
  contribute documentation with code, please insert your functional script
  into one of the two:

  * ``integ-tests.yml`` is run as part of our CI system. Code should run
    for no more than 30 seconds. It must not depend on data loaded from
    elsewhere, and not make use of surrogate blackboxes. It must not
    use SageMaker.
  * ``end-to-end-tests.yml`` is run manually on a regular basis, and in
    particular before a new release. Code may download files or depend on
    surrogate blackboxes. It may use SageMaker. Costs and runtime should
    be kept reasonable.

* Links to other parts of the documentation should be used frequently. We
  use anonymous references (two trailing underscores).
* Whenever mentioning a code construction (class, method, function, module,
  constant), please use a proper link with absolute module name and leading
  tilde. This allows interested readers to inspect API details and the code.
  When the same name is used several times in the same paragraph, it is
  sufficient to use a proper link for the first occurence only.
