Visualization Results of a Single Experiment
============================================

In this section, we describe the setup to be used for this tutorial. Then,
we show how the results of a single experiment can be visualized.

.. note::
   This tutorial shares some content with
   `this one <../benchmarking/bm_plotting.html>`__, but is more
   comprehensive in terms of features.

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

A Comparative Study
-------------------

For the purpose of this tutorial, we ran the setup of
`benchmarking/examples/benchmark_hypertune/ <../../benchmarking/benchmark_hypertune.html>`__,
using 15 random repetitions (or seeds). This is the command:

.. code-block:: bash

   python benchmarking/examples/benchmark_hypertune/launch_remote.py \
     --experiment_tag docs-1 --random_seed 2965402734 --num_seeds 15

Note that we fix the seed here in order to obtain repeatable results. Recall
from `here <../benchmarking/bm_simulator.html#defining-the-experiment>`__ that we compare 7
methods on 12 surrogate benchmarks:

* Since 4 of the 7 methods are "expensive", the above command launches
  ``3 + 4 * 15 = 63`` remote tuning jobs in parallel. Each of these jobs runs
  experiments for one method and all 12 benchmarks. For the "expensive" methods,
  each job runs a single seed, while for the remaining methods (ASHA, SYNCHB,
  BOHB), all seeds are run sequentially in a single job, so that a job for a
  "cheap" method runs ``12 * 15 = 180`` experiments sequentially.
* The total number of experiment runs is ``7 * 12 * 15 = 1260``
* Results of these experiments are stored to S3, using paths such as
  ``<s3-root>/syne-tune/docs-1/ASHA/docs-1-<datetime>/`` for ASHA (all seeds),
  or ``<s3-root>/syne-tune/docs-1/HYPERTUNE-INDEP-5/docs-1-<datetime>/`` for
  seed 5 of HYPERTUNE-INDEP. Result files are ``metadata.json``,
  ``results.csv.gz``, and ``tuner.dill``. The former two are required for plotting
  results.

Once all of this has finished, we are left with 3780 result files on S3. First,
we need to download the results from S3 to the local disk. This can be done by
a command which is also printed at the end of ``launch_remote.py``:

.. code-block:: bash

   aws s3 sync s3://<BUCKET-NAME>/syne-tune/docs-1/ ~/syne-tune/docs-1/ \
     --exclude "*" --include "*metadata.json" --include "*results.csv.zip"

This command can also be run from inside the plotting code. Note that the
``tuner.dill`` result files are not downloaded, since they are not needed for
result visualization.

Visualization of a Single Experiment
------------------------------------

For a single experiment, we can directly plot the best metric value obtained
as a function of wall-clock time. This can be done directly following the
experiment, as shown in
`this example <../../examples.html#plot-results-of-tuning-experiment>`__. In
our setup, experiments have been launched remotely, so in order to plot
results for a single experiment, we need to know the full tuner name.
Say, we would like to plot results of ``MOBSTER-JOINT``, ``seed=0``. The
names of single experiments are obtained by:

.. code-block:: bash

   ls ~/syne-tune/docs-1/MOBSTER-JOINT-0/

There is one experiment per benchmark, starting with ``docs-1-nas201-ImageNet16-120-0``,
``docs-1-nas201-cifar100-0``, ``docs-1-nas201-cifar10-0``, followed by date-time
strings. Once the tuner name is known, the following scripts plots the
desired curve and also displays the best configuration found:

.. literalinclude:: code/plot_single_experiment_results.py
   :caption: code/plot_single_experiment_results.py
   :start-after: # permissions and limitations under the License.

In general, you will have run more than one experiment. As in our study above,
you may want to compare different methods, or variations of the tuning problem.
You may want to draw conclusions by running on several benchmarks, and counter
random effects by repeating experiments several times. In the next section, we
show how comparative plots over many experiments can be created.
