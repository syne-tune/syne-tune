My Code Contains Packages
=========================

All code in :mod:`benchmarking.examples.demo_experiment` is contained in a single
directory. If your code for launching experiments and defining benchmarks is
structured into packages, you need to follow some extra steps.

There are two choices you have:

* Either, you
  `install Syne Tune from source <../../getting_started.html#installation>`__.
  In this case, you can just keep your launcher scripts and benchmark
  definitions in there, and use absolute imports from :mod:`benchmarking`.
  One advantage of this is that you can use all benchmarks currently included
  in :mod:`benchmarking.benchmark_definitions`.
* Or you do not install Syne Tune from source, in which case this section is for
  you.

We will use the example in :mod:`benchmarking.nursery.odsc_tutorial`. More
details about this example are found in
`this tutorial <../odsc_tutorial/README.html>`__. We will *not* assume that Syne
Tune is installed from source, but just that the code from
:mod:`benchmarking.nursery.odsc_tutorial` is present at ``<abspath>/odsc_tutorial/``.

The root package for this example is ``transformer_wikitext2``, in that all
imports start from there, for example:

.. literalinclude:: ../../../../benchmarking/nursery/odsc_tutorial/transformer_wikitext2/local/hpo_main.py
   :caption: transformer_wikitext2/local/hpo_main.py
   :start-after: # permissions and limitations under the License.

The code has the following structure:

.. code-block:: bash

   tree transformer_wikitext2/
   transformer_wikitext2
   ├── __init__.py
   ├── baselines.py
   ├── benchmark_definitions.py
   ├── code
   │   ├── __init__.py
   │   ├── requirements.txt
   │   ├── training_script.py
   │   ├── training_script_no_checkpoints.py
   │   ├── training_script_report_end.py
   │   └── transformer_wikitext2_definition.py
   ├── local
   │   ├── __init__.py
   │   ├── hpo_main.py
   │   ├── launch_remote.py
   │   ├── plot_learning_curve_pairs.py
   │   ├── plot_learning_curves.py
   │   ├── plot_results.py
   │   └── requirements-synetune.txt
   └── sagemaker
       ├── __init__.py
       ├── hpo_main.py
       ├── launch_remote.py
       ├── plot_results.py
       └── requirements.txt

Training code and benchmark definition are in ``code``, launcher and plotting
scripts for the local backend in ``local``, and ditto for the SageMaker backend
in ``sagemaker``.

In order to run any of the scripts, the ``PYTHONPATH`` environment variable needs
to be appended to as follows:

.. code-block:: bash

   export PYTHONPATH="${PYTHONPATH}:<abspath>/odsc_tutorial/"

Here, you need to replace ``<abspath>`` with the absolute path to ``odsc_tutorial``.
Once this is done, the following should work:

.. code-block:: bash

   python transformer_wikitext2/local/hpo_main.py \
     --experiment_tag mydebug --benchmark transformer_wikitext2 --num_seeds 1

Of course, this script needs all training script dependencies to be installed
locally. If you work with SageMaker, it is much simpler to launch experiments
remotely. The launcher script is as follows:

.. literalinclude:: ../../../../benchmarking/nursery/odsc_tutorial/transformer_wikitext2/local/launch_remote.py
   :caption: transformer_wikitext2/local/launch_remote.py
   :start-after: # permissions and limitations under the License.

Importantly, you need to set ``source_dependencies`` in this script. Here,
:code:`source_dependencies = [str(Path(__file__).parent.parent)]` translates
to :code:`["<abspath>/odsc_tutorial/transformer_wikitext2"]`. If you have
multiple root packages you want to import from, ``source_dependencies`` must
contain all of them.

The following command should work now:

.. code-block:: bash

   python transformer_wikitext2/local/launch_remote.py \
     --experiment_tag mydebug --benchmark transformer_wikitext2 --num_seeds 1 \
     --method BO

This should launch one SageMaker training job, which runs Bayesian optimization
with 4 workers. You can also test remote launching with the SageMaker backend:

.. code-block:: bash

   python transformer_wikitext2/sagemaker/launch_remote.py \
     --experiment_tag mydebug --benchmark transformer_wikitext2 --num_seeds 1 \
     --method BO --n_workers 2

This command should launch one SageMaker training job running Bayesian
optimization with the SageMaker backend, meaning that at any given time,
two worker training jobs are running.
