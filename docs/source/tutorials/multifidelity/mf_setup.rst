Setting up the Problem
======================

If you have not done this before, it is recommended you first work through the
`Basics of Syne Tune <../basics/README.html>`__ tutorial, in order to become
familiar with concepts such as *configuration*, *configuration space*,
*backend*, *scheduler*.

.. note::
   In this tutorial, we will use a surrogate benchmark in order to obtain
   realistic results with little computation. To this end, you need
   to have the ``blackbox-repository`` dependencies installed, as detailed
   `here <../faq.html#what-are-the-different-installations-options-supported>`__.
   Note that
   the first time you use a surrogate benchmark, its data files are downloaded
   and stored to your S3 bucket, this can take a considerable amount of time.
   The next time you use the benchmark, it is loaded from your local disk or
   your S3 bucket, which is fast.

Running Example
---------------

For most of this tutorial, we will be concerned with one running example: the
`NASBench-201 benchmark <https://arxiv.org/abs/2001.00326>`__. NASBench-201 is
a frequently used neural architecture search benchmark with a configuration
space of six categorical parameters, with five values each. The authors
trained networks under all these configurations and provide metrics, such as
training error, evaluation error and runtime after each epoch, free for
researchers to use. In this tutorial, we make use of the *CIFAR100* variant of
this benchmark, where the model architectures have been trained on the
*CIFAR100* image classification dataset.

NASBench-201 is an example for a *tabulated benchmark*. Researchers can
benchmark and compare HPO algorithms on the data without having to spend
efforts to train models. They do not need expensive GPU computation in order to
explore ideas or do comparative studies.

Syne Tune is particularly well suited to work with tabulated benchmarks. First,
it contains a blackbox repository for maintenance and fast access to tabulated
benchmarks. Second, it features a simulator backend which simulates training
evaluations from a blackbox. The simulator backend can be used with any Syne
Tune scheduler, and experiment runs are very close to what would be obtained by
running training for real. In particular, the simulation maintains correct
timings and temporal order of events. Importantly, time is simulated as well.
Not only are experiments very cheap to run (on basic CPU hardware), they also
finish many times faster than real time.

The Launcher Script
-------------------

The most flexible way to run HPO experiments in Syne Tune is by writing a
*launcher script*. In this tutorial, we will use the following launcher
script.

.. literalinclude:: code/hpo_main.py
   :caption: hpo_main.py
   :start-after: # permissions and limitations under the License.

Let us have a walk through this script, assuming it is called with the default
``--method ASHA-STOP``:

* If you worked through `Basics of Syne Tune <../basics/README.html>`__, you
  probably miss the training scripts. Since we use the simulator backend with a
  blackbox (NASBench-201), a training script is not required, since the backend
  is directly linked to the blackbox repository and obtains evaluation data from
  there.
* [1] We first select the benchmark and create the simulator backend linked
  with this benchmark. Relevant properties of supported benchmarks are
  collected in :mod:`syne_tune.experiments.benchmark_definitions`, using
  :class:`~syne_tune.experiments.benchmark_definitions.SurrogateBenchmarkDefinition`.
  Some properties are tied to the benchmark and must not be changed
  (``elapsed_time_attr``, ``metric``, ``mode``, ``blackbox_name``,
  ``max_resource_attr``). Other properties are default values suggested for the
  benchmark and may be changed by the user (``n_workers``,
  ``max_num_evaluations``, ``max_wallclock_time``, ``surrogate``). Some of the
  blackboxes are not computed on a dense grid, they require a surrogate
  regression model in order to be functional. For such, ``surrogate`` and
  ``surrogate_kwargs`` need to be considered. However, NASBench-201 comes with
  a finite configuration space, which has been sampled exhaustively.
* [1] We then create the
  :class:`~syne_tune.blackbox_repository.simulated_tabular_backend.BlackboxRepositoryBackend`.
  Instead of a training script, this backend needs information about the
  blackbox used for the simulation. ``elapsed_time_attr`` is the name of the
  *elapsed time* metric of the blackbox (time from start of training until end
  of epoch). ``max_resource_attr`` is the name of the maximum resource entry in
  the configuration space (more on this shortly).
* [2] Next, we select the configuration space and determine some attribute
  names. With a tabulated benchmark, we are bound to use the configuration space
  coming with the blackbox, ``trial_backend.blackbox.configuration_space``. If
  another configuration space is to be used, a surrogate regression model has to
  be specified. In this case, ``config_space_surrogate`` can be passed at the
  construction of ``BlackboxRepositoryBackend``. Since NASBench-201 has a native
  finite configuration space, we can ignore this extra complexity in this
  tutorial. However, choosing a suitable configuration space and specifying a
  surrogate can be important for model-based HPO methods. Some more informations
  are given `here <../../search_space.html>`__.
* [2] We can determine ``resource_attr`` (name of resource attribute) from the
  blackbox as ``blackbox.fidelity_name()``. Next, if ``max_resource_attr`` is
  specified, we attach the information about the largest resource level to the
  configuration space, via
  ``blackbox.configuration_space_with_max_resource_attr(max_resource_attr)``.
  Doing so is *best practice* in general. In the end, the training script needs
  to know how long to train for at most (i.e., the maximum number of epochs in
  our example), this should not be hardcoded. Another advantage of attaching
  the maximum resource information to the configuration space is that
  pause-and-resume schedulers can use it to signal the training script how long
  to really run for. This is explained in more detail
  `when we come to these schedulers <mf_asha.html#asynchronous-successive-halving-promotion-variant>`__.
  In short, we strongly recommend to use ``max_resource_attr`` and to configure
  schedulers with it.
* [2] If ``max_resource_attr`` is not to be used, the scheduler needs to be
  passed the maximum resource value explicitly. For ``ASHA-STOP``, this is the
  ``max_t`` attribute. This is not recommended, and not shown here.
* [3] At this point, we create the multi-fidelity scheduler, which is ASHA in
  the default case. Most supported schedulers can easily be imported from
  :mod:`syne_tune.optimizer.baselines`, using common names.
* [4] Finally, we create a stopping criterion and a ``Tuner``. This should be
  well known from `Basics of Syne Tune <../basics/README.html>`__. One
  speciality here is that we require ``sleep_time=0`` and
  ``callbacks=[SimulatorCallback()]`` for things to work out with the simulator
  backend. Namely, since time is simulated, the ``Tuner`` does not really have
  to sleep between its iterations (simulated time will be increased in distinct
  steps). Second,
  :class:`~syne_tune.backend.simulator_backend.simulator_callback.SimulatorCallback`
  is needed for simulation of time. It is fine to add additional callbacks here,
  as long as ``SimulatorCallback`` is one of them.

The Blackbox Repository
-----------------------

Giving a detailed account of the blackbox repository is out of scope of this
tutorial. If you run the launcher script above, you will be surprised how
quickly it finishes. The only real time spent is on logging, fetching metric
values from the blackbox, and running the scheduler code. Since the latter is
very fast (mostly some random sampling and data organization), whole simulated
HPO experiments with many parallel workers can be done in mere seconds.

However, when you run it for the very first time, you will have to wait for
quite some time. This is because the blackbox repository downloads the raw data
for the benchmark of your choice, processes it, and (optionally) stores it to
your S3 bucket. It also stores a local copy. If the data is already in your S3
bucket, it will be downloaded from there if you run on a different instance,
this is rather fast. But downloading and processing the raw data can take an
hour or more for some of the blackboxes.
