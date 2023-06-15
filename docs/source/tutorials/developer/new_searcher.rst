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
`FIFOScheduler <random_search.html#fifoscheduler-and-randomsearcher>`__ or
`HyperbandScheduler <extend_async_hb.html#hyperbandscheduler>`__ allow the
user to choose a searcher via the string argument ``searcher``, and to
configure the searcher (away from defaults) by the dictionary argument
``search_options``. While ``searcher`` can also be a
`BaseSearcher <random_search.html#fifoscheduler-and-randomsearcher>`__
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
`shown here <random_search.html#fifoscheduler-and-randomsearcher>`__). Here are
some best practices for linking a new searcher into the factory:

* The Syne Tune code is written in a way which allows to run certain scenarios
  with a restricted set of all possible dependencies (see
  `FAQ <../../faq.html#what-are-the-different-installations-options-supported>`__).
  This is achieved by conditional imports. If your searcher requires
  dependencies beyond the core, please make sure to use
  ``try ... except ImportError`` as you see in the code.
* Try to make sure that your searcher also works without ``search_options``
  being specified by the user. You will always have the fields contributed by
  the generic schedulers, and for all others, your code should ideally come with
  sensible defaults.
* Make sure to implement the ``configure_scheduler`` method of your new searcher,
  restricting usage to supported scheduler types.

The Baseline Wrappers
---------------------

In order to facilitate choosing and configuring a scheduler along with its
searcher, Syne Tune defines the most frequently used combinations in
:mod:`syne_tune.optimizer.baselines`. The minimal signature of a baseline
class is this:

.. code-block:: python

   def __init__(self, config_space: dict, metric: str, **kwargs):

Or, in the multi-objective case:

.. code-block:: python

   def __init__(self, config_space: dict, metric: List[str], **kwargs):

If the underlying scheduler maintains a searcher (as most schedulers do),
arguments to the searcher (except for ``config_space``, ``metric``) are
given in ``kwargs["search_options"]``. If a scheduler is of multi-fidelity
type, the minimal signature is:

.. code-block:: python

   def __init__(self, config_space: dict, metric: str, resource_attr: str, **kwargs):

If the scheduler accepts a random seed, this must be ``kwargs["random_seed"]``.
Several wrapper classes in :mod:`syne_tune.optimizer.baselines` have signatures
with more arguments, which are either passed to the scheduler or to the searcher.
For example, some wrappers make ``random_seed`` explicit in the signature,
instead of having it in ``kwargs``.

.. note::
   If a scheduler maintains a searcher inside, and in particular if it simply
   configures :class:`~syne_tune.optimizer.schedulers.FIFOScheduler` or
   class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` with a new
   searcher, it is strongly recommended to adhere to the policy to specify
   searcher arguments in ``kwargs["search_options"]``. This simplifies enabling
   the new scheduler in the simple experimentation framework of
   :mod:`syne_tune.experiments`, and in general provides a common user
   experience across different schedulers.

Let us look at an example of a baseline wrapper whose underlying scheduler is
of type :class:`~syne_tune.optimizer.schedulers.FIFOScheduler` with a specific
searcher, which is not itself created via a searcher factory:

.. literalinclude:: ../../../../syne_tune/optimizer/baselines.py
   :caption: syne_tune/optimizer/baselines.py -- REA
   :start-at: class REA(FIFOScheduler):
   :end-before: class MOREA(FIFOScheduler):

* The signature has ``config_space``, ``metric``, and ``random_seed``. It also
  has two searcher arguments, ``population_size`` and ``sample_size``.
* In order to compile the arguments ``searcher_kwargs`` for creating the
  searcher, we first call
  :code:`_create_searcher_kwargs(config_space, metric, random_seed, kwargs)`.
  Doing so is particularly important in order to ensure random seeds are
  managed between scheduler and searcher in the same way across different
  Syne Tune schedulers.
* Next, the additional arguments ``population_size`` and ``sample_size`` need
  to be appended to these searcher arguments. Had we used
  ``kwargs["search_options"]`` instead, this would not be necessary.
* Finally, we create :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`,
  passing ``config_space``, ``metric``, as well as the new searcher via
  :code:`searcher=RegularizedEvolution(**searcher_kwargs)`, and finally pass
  ``**kwargs`` at the end.

Baselines and Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~

As shown in `this tutorial <../experimentation/README.html>`__ and
`this tutorial <../odsc_tutorial/README.html>`__, a particularly convenient
way to define and run experiments is using the code in
:mod:`syne_tune.experiments`. Once a new scheduler has a baseline wrapper, it
is very easy to make it available there: you just need to add a wrapper in
:mod:`syne_tune.experiments.default_baselines`. For the ``REA`` example above,
this is:

.. code-block:: python

   from syne_tune.optimizer.baselines import REA as _REA

   def REA(method_arguments: MethodArguments, **kwargs):
       return _REA(**_baseline_kwargs(method_arguments, kwargs))

Contribute your Extension
-------------------------

At this point, you are ready to plug in your latest idea and make it work in
Syne Tune. Given that it works well, we would encourage you to
`contribute it back to the community <https://github.com/awslabs/syne-tune/blob/main/CONTRIBUTING.md>`__.
We are looking forward to your pull request.
