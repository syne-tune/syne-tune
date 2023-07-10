Setting up an Experimental Study
================================

Any statistical analysis consists of a sequence of experiments, where later
ones are planned given outcomes of earlier ones. Parallelization can be used
to speed up this process:

* If outcomes or decision-making are randomized (e.g., training neural networks
  starts from random initial weights; HPO may suggest configurations drawn at
  random), it is important to repeat experiments several times in order to
  gain robust outcomes.
* If a search problem becomes too big, it can be broken down into several parts,
  which can be worked on independently.

In this section, we describe the setup for a simple study, which can be used
to showcase tooling in Syne Tune for splitting up a large problem into pieces,
running random repetitions, writing out extra information, and creating
customized comparative plots.

For simplicity, we use surrogate benchmarks from the ``fcnet`` family, whereby
tuning is `simulated <../benchmarking/bm_simulator.html>`__. This is the
default configuration space for these benchmarks:

.. literalinclude:: ../../../../syne_tune/blackbox_repository/conversion_scripts/scripts/fcnet_import.py
   :caption: syne_tune/blackbox_repository/conversion_scripts/scripts/fcnet_import.py
   :start-at: CONFIGURATION_SPACE = {
   :end-before: def convert_dataset(dataset_path: Path, max_rows: int = None):

.. note::
   In the Syne Tune experimentation framework, a tuning problem (i.e., training and
   evaluation script or blackbox, together with defaults) is called a *benchmark*.
   This terminology is used even if the goal of experimentation is not benchmarking
   (i.e., comparing different HPO methods), as is the case in this tutorial here.

.. note::
   The code used in this tutorial is contained in the
   `Syne Tune source <../../getting_started.html#installation>`__, it is not
   installed by ``pip``. You can obtain this code by installing Syne Tune from
   source, but the only code that is needed is in
   :mod:`benchmarking.examples.demo_experiment`, so if you copy that out of the
   repository, you do not need all the remaining source code.

.. note::
   In order to use surrogate benchmarks and the simulator backend, you need
   to have the ``blackbox-repository`` dependencies installed, as detailed
   `here <../faq.html#what-are-the-different-installations-options-supported>`__.
   Note that the first time you use a surrogate benchmark, its data files are
   downloaded and stored to your S3 bucket, this can take a considerable amount
   of time. The next time you use the benchmark, it is loaded from your local
   disk or your S3 bucket, which is fast.

Modifying the Configuration Space
---------------------------------

The hyperparameters ``hp_activation_fn_1`` and ``hp_activation_fn_2`` prescribe
the type of activation function in hidden layers 1 and 2. We can split the
overall tuning problem into smaller pieces by fixing these parameters to
fixed values, considering ``relu`` and ``tanh`` networks independently. In our
study, we will compare the following methods:

* ``ASHA-TANH``, ``MOBSTER-TANH``: Runs ASHA and MOBSTER on the
  simplified configuration space, where
  ``hp_activation_fn_1 = hp_activation_fn_2 = "tanh"``
* ``ASHA-RELU``, ``MOBSTER-RELU``: Runs ASHA and MOBSTER on the
  simplified configuration space, where
  ``hp_activation_fn_1 = hp_activation_fn_2 = "relu"``
* ``ASHA``, ``MOBSTER``: Runs ASHA and MOBSTER on the original
  configuration space
* ``RS``, ``BO``: Runs baselines random search and Bayesian optimization on
  the original configuration space

Here is the script defining these alternatives:

.. literalinclude:: ../../../../benchmarking/examples/demo_experiment/baselines.py
   :caption: benchmarking/examples/demo_experiment/baselines.py
   :start-after: # permissions and limitations under the License.

* Different methods are defined in dictionary ``methods``, as functions
  mapping ``method_arguments`` of type
  :class:`~syne_tune.experiments.baselines.MethodArguments` to a scheduler
  object. Here, ``method_arguments.config_space`` contains the default
  configuration space for the benchmark, where both
  ``hp_activation_fn_1`` and ``hp_activation_fn_2`` are hyperparameters
  of type ``choice(["tanh", "relu"])``.
* For ``ASHA-TANH``, ``MOBSTER-TANH``, ``ASHA-RELU``, ``MOBSTER-RELU``, we fix
  these parameters. This is done in ``_modify_config_space``, where
  ``method_arguments.config_space`` is replaced by a configuration space where
  the two hyperparameters are fixed (so methods do not search over them
  anymore).
* Another way to modify ``method_arguments`` just before a method is created,
  is to use the ``map_extra_args`` argument of
  :func:`~syne_tune.experiments.launchers.hpo_main_simulator.main`, as detailed
  `here <../benchmarking/bm_simulator.html#specifying-extra-arguments>`__. This
  allows the modification to depend on extra command line arguments.

Next, we define the benchmarks our study should run over. For our simple
example, we use the ``fcnet`` benchmarks:

.. literalinclude:: ../../../../benchmarking/examples/demo_experiment/benchmark_definitions.py
   :caption: benchmarking/examples/demo_experiment/benchmark_definitions.py
   :start-after: # permissions and limitations under the License.

This is where you would have to plug in your own benchmarks, namely your training
script with a bit of metadata. Examples are provided
`here <../benchmarking/bm_local.html>`__ and
`here <../benchmarking/bm_contributing.html>`__.

Recording Extra Results
-----------------------

Next, we need to write the ``hpo_main.py`` script which runs a single experiment.
As shown `here <../benchmarking/bm_simulator.html#defining-the-experiment>`__,
this is mostly about selecting the correct ``main`` function among
:func:`syne_tune.experiments.launchers.hpo_main_simulator.main`,
:func:`syne_tune.experiments.launchers.hpo_main_local.main`,
:func:`syne_tune.experiments.launchers.hpo_main_sagemaker.main`, depending on the trial
backend we want to use. In our case, we also would like to record extra
information about the experiment. Here is the script:

.. literalinclude:: ../../../../benchmarking/examples/demo_experiment/hpo_main.py
   :caption: benchmarking/examples/demo_experiment/hpo_main.py
   :start-after: # permissions and limitations under the License.

* As usual, we import :func:`syne_tune.experiments.launchers.hpo_main_simulator.main`
  (we use the simulator backend) and call it, passing our ``methods`` and
  ``benchmark_definitions``. We also pass ``extra_results``, since we would
  like to record extra results.
* Note that apart from :mod:`syne_tune` imports, this script is only doing local
  imports. No other code from :mod:`benchmarking` is required.
* A certain number of time-stamped results are recorded by default in
  ``results.csv.zip``, details are
  `here <../../faq.html#what-does-the-output-of-the-tuning-contain>`__. In
  particular, all metric values reported for all trials are recorded.
* In our example, we would also like to record information about the
  multi-fidelity schedulers ASHA and MOBSTER. As detailed in
  `this tutorial <../multifidelity/mf_asha.html>`__, they record metric
  values for trials at different rung levels these trials reached (e.g.,
  number of epochs trained), and decisions on which paused trial to
  promote to the next rung level are made by comparing its performance with
  all others in the same rung. The rung levels are growing over time, and
  we would like to record their respective sizes as a function of wall-clock
  time.
* To this end, we create a subclass of
  :class:`~syne_tune.results_callback.ExtraResultsComposer`, whose
  ``__call__`` method extracts the desired information from the current
  :class:`~syne_tune.Tuner` object. In our example, we first test whether
  the current scheduler is ASHA or MOBSTER (recall that we also run
  ``RS`` and ``BO`` as baselines). If so, we extract the desired information
  and return it as a dictionary.
* Finally, we create ``extra_results`` and pass it to the ``main`` function.

The outcome is that a number of additional columns are appended to the dataframe
stored in ``results.csv.zip``, at least for experiments with ASHA or
MOBSTER schedulers. Running this script launches an experiment locally (if you
installed Syne Tune from source, you need to start the script from the
``benchmarking/examples`` directory):

.. code-block:: bash

   python demo_experiment/hpo_main.py --experiment_tag docs-2-debug

Running Experiments in Parallel
-------------------------------

Running our ``hpo_main.py`` script launches a single experiment on the local
machine, writing results to a local directory. This is nice for debugging, but
slow and cumbersome once we convinced ourselves that the setup is working. We
will want to launch many experiments in parallel on AWS, and use our local
machine for other work.

* Experiments with our setups ``RS``, ``BO``, ``ASHA-TANH``, ``MOBSTER-TANH``,
  ``ASHA-RELU``, ``MOBSTER-RELU``, ``ASHA``, ``MOBSTER`` are independent and
  can be run in parallel.
* We repeat each experiment 20 times, in order to quantify the random
  fluctuation in the results. These seeds are independent and can be run
  in parallel.
* We could also run experiments with different benchmarks (i.e., datasets in
  ``fcnet``) in parallel. But since a single simulated experiment is fast to
  do, we are not doing this here.

Running experiments in parallel requires a remote launcher script:

.. literalinclude:: ../../../../benchmarking/examples/demo_experiment/launch_remote.py
   :caption: benchmarking/examples/demo_experiment/launch_remote.py
   :start-after: # permissions and limitations under the License.

* Again, we simply choose the correct ``launch_remote`` function among
  :func:`~syne_tune.experiments.launchers.launch_remote_simulator.launch_remote`,
  :func:`~syne_tune.experiments.launchers.launch_remote_main.launch_remote`,
  :func:`~syne_tune.experiments.launchers.launch_remote_sagemaker.launch_remote`,
  depending on the trial backend.
* Note that apart from :mod:`syne_tune` imports, this script is only doing local
  imports. No other code from :mod:`benchmarking` is required.
* In ``is_expensive_method``, we pass a predicate from method name. If
  ``is_expensive_method(method)`` is ``True``, the 20 different seeds are
  run in parallel. Otherwise, they are run sequentially.
* In our example, we know that ``BO`` and ``MOBSTER`` run quite a bit slower
  in the simulator than ``RS`` and ``ASHA``, so we label the former as expensive.
  This means we have 4 expensive methods and 4 cheap ones, and our complete
  study will launch ``4 + 4 * 20 = 84`` SageMaker training jobs. Since
  ``fcnet`` contains four benchmarks, we run ``8 * 20 * 4 = 640`` experiments
  in total.

All of these experiments can be launched with a single command (if you
installed Syne Tune from source, you need to start the script from the
``benchmarking/examples`` directory):

.. code-block:: bash

   python demo_experiment/launch_remote.py \
     --experiment_tag docs-2 --random_seed 2465497701 --num_seeds 20

If ``--random_seed`` is not given, a master random seed is drawn at random,
printed and also stored in the metadata. If a study consists of launching
experiments in several steps, it is good practice to pass the same random seed
for each launch command. For example, you can run the first launch command
without passing a seed, then note the seed from the output and use it for
further launches.

Avoiding Costly Failures
~~~~~~~~~~~~~~~~~~~~~~~~

In practice, with a new experimental setup, it is not a good idea to launch
all experiments in one go. We recommend to move in stages.

First, if our benchmarks run locally as well, we should start with some local
tests. For example:

.. code-block:: bash

   python demo_experiment/hpo_main.py \
     --experiment_tag docs-2-debug --random_seed 2465497701 \
     --method ASHA-RELU --verbose 1

We can cycle through several methods and check whether anything breaks. Note that
``--verbose 1`` generates useful output about the progress of the method, which
can be used to check whether properties are the way we expect (for example,
``"relu"`` is chosen for the fixed hyperparameters). Results are stored locally
under ``~/syne_tune/docs-2-debug/``.

Next, we launch the setup remotely, but for a single seed:

.. code-block:: bash

   python demo_experiment/launch_remote.py \
     --experiment_tag docs-2 --random_seed 2465497701 --num_seeds 1

This will start 8 SageMaker training jobs, one for each method, and with
``seed=0``. Some of them, like ``RS``, ``ASHA``, ``ASHA-*`` will finish very
rapidly, and it makes sense to quickly browse their logs, to check whether
desired properties are met.

Finally, if this looks good, we can launch all the rest:

.. code-block:: bash

   python demo_experiment/launch_remote.py \
     --experiment_tag docs-2 --random_seed 2465497701 --num_seeds 20 \
     --start_seed 1

This is launching all remaining experiments with ``seed`` from 1 to 19.

.. note::
   If something breaks when remotely launching for ``seed=0``, it may be that
   results have already been written to S3. This is because results are written
   out periodically. If you use the same tag ``docs-2`` for initial debugging,
   you will have to remove these results on S3, or otherwise be careful filtering
   them out later on (this is discussed below).

In a large study consisting of many experiments, it can happen that some
experiments fail for reasons which do not invalidate results of the other ones.
If this happens, it is not a good idea, both time and cost wise, to start the
whole study from scratch. Instead, we recommend to clean up and restart only
the experiments which failed. For example, assume that in our study above,
the ``MOBSTER-TANH`` experiments of ``seed == 13`` failed:

* We need to remove incomplete results of these experiments, which can corrupt
  final aggregate results otherwise. This can either be done by removing them
  on S3, or by advanced filtering (discussed below). In general, we recommend
  the former. For our example, the results to be removed are in
  ``s3://{sagemaker-default-bucket}/syne-tune/docs-2/MOBSTER-TANH-13/``. Namely,
  since ``MOBSTER-TANH`` is an "expensive" method, results for different seeds
  are written to different subdirectories.
* Next, we need to start the failed experiments again:

.. code-block:: bash

   python demo_experiment/launch_remote.py \
     --experiment_tag docs-2 --random_seed 2465497701 --num_seeds 14 \
     --start_seed 13 --method MOBSTER-TANH

Instead, assume that the ``ASHA`` experiments for ``seed == 13`` failed. This is
a "cheap" method, so results for all seeds are written to
``s3://{sagemaker-default-bucket}/syne-tune/docs-2/ASHA/``, into subdirectories
of the form ``docs-2-<benchmark>-<seed>-<datetime>``. Since this method is cheap,
we can rerun all its experiments, by first removing everything under
``s3://{sagemaker-default-bucket}/syne-tune/docs-2/ASHA/``, then:

.. code-block:: bash

   python demo_experiment/launch_remote.py \
     --experiment_tag docs-2 --random_seed 2465497701 --num_seeds 20 \
     --method ASHA

.. note::
   Don't worry if you restart failed experiments without first removing its
   incomplete results on S3. Due to the ``<datetime>`` postfix of directory
   names, results of a restart never conflict with older ones. However, once
   you plot aggregate results, you will get a warning that too many results
   have been found, along with where these results are located. At this point,
   you can still remove the incomplete ones.
