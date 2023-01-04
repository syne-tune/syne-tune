Linking in a New Searcher
=========================

At this point, you should have learned everything needed for implementing a new
scheduler, or for modifying an existing template scheduler to your special
requirements. Say, you have implemented a new searcher to be plugged into one
of the existing generic schedulers. In this section, we will look into how a
new searcher can be made available in an easy-to-use fashion.

The Searcher Factory
--------------------

Recall that our generic schedulers, such as
`FIFOScheduler <random_search.html#fifoscheduler-and-randomsearcher>`_ or
`HyperbandScheduler <extend_async_hb.html#hyperbandscheduler>`_ allow the
user to choose a searcher via the string argument ``searcher``, and to
configure the searcher (away from defaults) by the dictionary argument
``search_options``. While ``searcher`` can also be a
`BaseSearcher <random_search.html#fifoscheduler-and-randomsearcher>`_
instance, it is simpler and more convenient to choose the searcher by
name. For example:

* Generic schedulers only work with certain types of searchers. This
  consistency is checked when ``searcher`` is a name, but may lead to subtle
  errors if not.
* Several arguments of a searcher are typically just the same as for the
  surrounding scheduler, or can be inferred from arguments of the scheduler.
  This can become complex for some searchers and leads to difficult boiler plate
  code in case ``searcher`` is to be created by hand.
* While not covered in this tutorial, constructing schedulers and searchers for
  Gaussian process based Bayesian optimization and its extensions to
  multi-fidelity scheduling, constrained or cost-aware search is significantly
  more complex, as can be seen in
  :mod:`syne_tune.optimizer.schedulers.searchers.gp_searcher_factory`.

It is the purpose of
:func:`~syne_tune.optimizer.schedulers.searchers.searcher_factory.searcher_factory`
to create the correct
:class:`~syne_tune.optimizer.schedulers.searchers.BaseSearcher` object for given
scheduler arguments, including ``searcher`` (name) and ``search_options``. Let
us have a look how the constructor of
:class:`~syne_tune.optimizer.schedulers.FIFOScheduler` calls the factory. We see
how scheduler arguments like ``metric``, ``mode``, ``points_to_evaluate`` are
just passed through to the factory. We also need to set
``search_options["scheduler"]`` in order to tell ``searcher_factory`` which
generic scheduler is calling it.

The
:func:`~syne_tune.optimizer.schedulers.searchers.searcher_factory.searcher_factory`
code should be straightforward to understand and extend. Pick a name for your
new searcher and set ``searcher_cls`` and ``supported_schedulers`` (the latter
can be left to ``None`` if your searcher works with all generic schedulers). The
constructor of your searcher needs to have the signature

.. code-block:: python

   def __init__(self, config_space: dict, metric: str, **kwargs):

Here, ``kwargs`` will be fed with ``search_options``, but enriched with fields
like ``mode``, ``points_to_evaluate``, ``random_seed_generator``, ``scheduler``.
Your searcher is not required to make use of them, even though we strongly
recommend to support ``points_to_evaluate`` and to make use of
``random_seed_generator`` (as is
`shown here <random_search.html#fifoscheduler-and-randomsearcher>`_). Here are
some best practices for linking a new searcher into the factory:

* The Syne Tune code is written in a way which allows to run certain scenarios
  with a restricted set of all possible dependencies (see
  `FAQ <../../faq.html#what-are-the-different-installations-options-supported>`_).
  This is achieved by conditional imports. If your searcher requires
  dependencies beyond the core, please make sure to use
  ``try ... except ImportError`` as you see in the code.
* Try to make sure that your searcher also works without ``search_options``
  being specified by the user. You will always have the fields contributed by
  the generic schedulers, and for all others, your code should ideally come with
  sensible defaults.
* Make sure to implement the ``configure_scheduler`` method of your new searcher,
  restricting usage to supported scheduler types.

Contribute your Extension
-------------------------

At this point, you are ready to plug in your latest idea and make it work in
Syne Tune. Given that it works well, we would encourage you to
`contribute it back to the community <https://github.com/awslabs/syne-tune/blob/main/CONTRIBUTING.md>`_.
We are looking forward to your pull request.
