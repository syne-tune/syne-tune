Launching Experiments Remotely
==============================

As a machine learning practitioner, you operate in a highly competitive
landscape. Your success depends to a large extent on whether you can *decrease
the time to the next decision*. In this section, we discuss one important
approach, namely how to increase the *number of experiments run in parallel*.

.. note::
   Imports in our scripts are absolute against the root package
   ``transformer_wikitext2``, so that only the code in
   :mod:`benchmarking.nursery.odsc_tutorial` has to be present. In order to run
   them, you need to append ``<abspath>/odsc_tutorial/`` to the ``PYTHONPATH``
   environment variable. This is required even if you have installed Syne Tune
   from source.

Launching our Study
-------------------

Here is how we specified and ran experiments of our
`study <comparison.html#a-comparative-study>`__. First, we specify a
script for launching experiments locally:

.. literalinclude:: ../../../../benchmarking/nursery/odsc_tutorial/transformer_wikitext2/local/hpo_main.py
   :caption: transformer_wikitext2/local/hpo_main.py
   :start-after: # permissions and limitations under the License.

This is very simple, as most work is done by the generic
:func:`syne_tune.experiments.launchers.hpo_main_local.main`. Note that ``hpo_main_local``
needs to be chosen, since we use the local backend.

This local launcher script can be used to configure your experiment, given
additional command line arguments, as is explained in detail
`here <../benchmarking/bm_simulator.html#specifying-extra-arguments>`__.

You can use ``hpo_main.py`` to launch experiments locally, but they'll run
sequentially, one after the other, and you need to have all dependencies
installed locally. A second script is needed in order to launch many
experiments in parallel:

.. literalinclude:: ../../../../benchmarking/nursery/odsc_tutorial/transformer_wikitext2/local/launch_remote.py
   :caption: transformer_wikitext2/local/launch_remote.py
   :start-after: # permissions and limitations under the License.

Once more, all the hard work in done in
:func:`syne_tune.experiments.launchers.launch_remote_local.launch_remote`, where
``launch_remote_local`` needs to be chosen for the local backend. Most important
is that our previous ``hpo_main.py`` is specified as ``entry_point`` here. Here is
the command to run all experiments of our study in parallel (replace ``...`` by the
absolute path to ``odsc_tutorial``):

.. code-block:: bash

   export PYTHONPATH="${PYTHONPATH}:/.../odsc_tutorial/"
   python transformer_wikitext2/local/launch_remote.py \
     --experiment_tag odsc-1 --benchmark transformer_wikitext2 --num_seeds 10

* This command launches 40 SageMaker training jobs, running 10 random repetitions
  (seeds) for each of the 4 methods specified in ``baselines.py``.
* Each SageMaker training job uses one ``ml.g4dn.12xlarge`` AWS instance. You can
  only run all 40 jobs in parallel if your resource limit for this instance type
  is 40 or larger. Each training job will run a little longer than 5 hours, as
  specified by ``max_wallclock_time``.
* You can use ``--instance_type`` and ``--max_wallclock_time`` command line
  arguments to change these defaults. However, if you choose an instance type with
  less than 4 GPUs, the local backend will not be able to run 4 trials in parallel.
* If ``benchmark_definitions.py`` defines a single benchmark only, the
  ``--benchmark`` argument can also be dropped.

When using remote launching, results of your experiments are written to S3, to
the default bucket for your AWS account. Once all jobs have finished (which takes
a little more than 5 hours if you have sufficient limits, and otherwise longer),
you can create the comparative plot shown
`above <comparison.html#a-comparative-study>`__, using this script:

.. literalinclude:: ../../../../benchmarking/nursery/odsc_tutorial/transformer_wikitext2/local/plot_results.py
   :caption: transformer_wikitext2/local/plot_results.py
   :start-after: # permissions and limitations under the License.

For details about visualization of results in Syne Tune, please consider
`this tutorial <../visualization/README.html>`__. In a nutshell, this is what
happens:

* Collect and filter results from all experiments of a study
* Group them according to setup (HPO method here), aggregate over seeds
* Create plot in which each setup is represented by a curve and confidence bars
