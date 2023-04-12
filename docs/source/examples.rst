Launch HPO Experiment Locally
=============================

.. literalinclude:: ../../examples/launch_height_baselines.py
   :caption: examples/launch_height_baselines.py
   :start-after: # permissions and limitations under the License.

Along with several of the examples below, this launcher script is using the
following :ref:`train_height.py <train_height_script>` training script:

.. literalinclude:: ../../examples/training_scripts/height_example/train_height.py
   :name: train_height_script
   :caption: examples/training_scripts/height_example/train_height.py
   :start-after: # permissions and limitations under the License.


Fine-Tuning Hugging Face Model for Sentiment Classification
===========================================================

.. literalinclude:: ../../examples/launch_huggingface_classification.py
   :caption: examples/launch_huggingface_classification.py
   :start-after: # permissions and limitations under the License.

**Requirements**:

* Needs ``benchmarking`` framework, which requires Syne Tune to be installed
  `from source <getting_started.html#installation>`__.
* `Access to AWS SageMaker <faq.html#how-can-i-run-on-aws-and-sagemaker>`__
* Runs on four ``ml.g4dn.xlarge`` instances

In this example, we use the SageMaker backend together with the
SageMaker Hugging Face framework in order to fine-tune a DistilBERT
model on the IMDB sentiment classification task. This task is one of
our built-in benchmarks. For other ways to run this benchmark on
different backends or remotely, consult
`this tutorial <tutorials/benchmarking/README.html>`__.

A more advanced example for fine-tuning Hugging Face transformers is given
`here <benchmarking/fine_tuning_transformer_glue.html>`__.


Launch HPO Experiment with Python Backend
=========================================

.. literalinclude:: ../../examples/launch_height_python_backend.py
   :caption: examples/launch_height_python_backend.py
   :start-after: # permissions and limitations under the License.

The Python backend does not need a separate training script.


Population-Based Training (PBT)
===============================

.. literalinclude:: ../../examples/launch_pbt.py
   :caption: examples/launch_pbt.py
   :start-after: # permissions and limitations under the License.

This launcher script is using the following :ref:`pbt_example.py <pbt_example_script>` training
script:

.. literalinclude:: ../../examples/training_scripts/pbt_example/pbt_example.py
   :name: pbt_example_script
   :caption: examples/training_scripts/pbt_example/pbt_example.py
   :start-after: # permissions and limitations under the License.

For this toy example, PBT is run with a population size of 2, so only
two parallel workers are needed. In order to use PBT competitively,
choose the SageMaker backend. Note that PBT requires your training
script to
`support checkpointing <faq.html#how-can-i-enable-trial-checkpointing>`__.


Visualize Tuning Progress with Tensorboard
==========================================

.. literalinclude:: ../../examples/launch_tensorboard_example.py
   :caption: examples/launch_tensorboard_example.py
   :start-after: # permissions and limitations under the License.

**Requirements**:

* Needs ``tensorboardX`` to be installed: ``pip install tensorboardX``.

Makes use of :ref:`train_height.py <train_height_script>`.

Tensorboard visualization works by using a callback, for example
:class:`~syne_tune.callbacks.tensorboard_callback.TensorboardCallback`,
which is passed to the :class:`~syne_tune.Tuner`. In order to visualize
other metrics, you may have to modify this callback.


Launch HPO Experiment with Simulator Backend
============================================

.. literalinclude:: ../../examples/launch_nasbench201_simulated.py
   :caption: examples/launch_nasbench201_simulated.py
   :start-after: # permissions and limitations under the License.

**Requirements**:

* Needs ``benchmarking`` framework, which requires Syne Tune to be installed
  `from source <getting_started.html#installation>`__.
* Needs ``nasbench201`` blackbox to be downloaded and preprocessed. This can
  take quite a while when done for the first time
* If `AWS SageMaker is used  <faq.html#how-can-i-run-on-aws-and-sagemaker>`__
  or an S3 bucket is accessible, the blackbox files are uploaded to your S3
  bucket

In this example, we use the simulator backend with the NASBench-201
blackbox. Since time is simulated, we can use
``max_wallclock_time=3600`` (one hour), but the experiment finishes
in mere seconds. More details about the simulator backend is found in
`this tutorial <tutorials/benchmarking/bm_simulator.html>`__.


Joint Tuning of Instance Type and Hyperparameters using MOASHA
==============================================================

.. literalinclude:: ../../examples/launch_moasha_instance_tuning.py
   :caption: examples/launch_moasha_instance_tuning.py
   :start-after: # permissions and limitations under the License.

**Requirements**:

* Needs ``benchmarking`` framework, which requires Syne Tune to be installed
  `from source <getting_started.html#installation>`__.
* `Access to AWS SageMaker <faq.html#how-can-i-run-on-aws-and-sagemaker>`__
* Runs training jobs on instances of type ``ml.g4dn.xlarge``, ``ml.g5.xlarge``,
  ``ml.g4dn.2xlarge``, ``ml.p2.xlarge``, ``ml.g5.2xlarge``, ``ml.g5.4xlarge``,
  ``ml.g4dn.4xlarge``, ``ml.g5.8xlarge``, ``ml.g4dn.8xlarge``,
  ``ml.p3.2xlarge``, ``ml.g5.16xlarge``. This list of instances types to be
  searched over can be modified by the user

In this example, we use the SageMaker backend together with the
SageMaker Hugging Face framework in order to fine-tune a DistilBERT
model on the IMDB sentiment classification task:

* Instead of optimizing a single objective, we use
  :class:`~syne_tune.optimizer.schedulers.multiobjective.MOASHA` in order
  to sample the Pareto frontier w.r.t. three objectives
* We not only tune hyperparameters such as learning rate and weight
  decay, but also the AWS instance type to be used for training. Here,
  one of the objectives to minimize is the training cost (in dollars).


Multi-objective Asynchronous Successive Halving (MOASHA)
========================================================

.. literalinclude:: ../../examples/launch_height_moasha.py
   :caption: examples/launch_height_moasha.py
   :start-after: # permissions and limitations under the License.

This launcher script is using the following :ref:`mo_artificial.py <mo_artificial_script>` training
script:

.. literalinclude:: ../../examples/training_scripts/mo_artificial/mo_artificial.py
   :name: mo_artificial_script
   :caption: examples/training_scripts/mo_artificial/mo_artificial.py
   :start-after: # permissions and limitations under the License.


PASHA: Efficient HPO and NAS with Progressive Resource Allocation
=================================================================

.. literalinclude:: ../../examples/launch_pasha_nasbench201.py
   :caption: examples/launch_pasha_nasbench201.py
   :start-after: # permissions and limitations under the License.

**Requirements**:

* Needs ``benchmarking`` framework, which requires Syne Tune to be installed
  `from source <getting_started.html#installation>`__.
* Needs ``nasbench201`` blackbox to be downloaded and preprocessed. This can
  take quite a while when done for the first time

PASHA typically uses ``max_num_trials_completed`` as the stopping criterion.
After finding a strong configuration using PASHA, 
the next step is to fully train a model with the configuration.


Constrained Bayesian Optimization
=================================

.. literalinclude:: ../../examples/launch_bayesopt_constrained.py
   :caption: examples/launch_bayesopt_constrained.py
   :start-after: # permissions and limitations under the License.

This launcher script is using the following :ref:`train_constrained_example.py <train_constrained_script>` training
script:

.. literalinclude:: ../../examples/training_scripts/constrained_hpo/train_constrained_example.py
   :name: train_constrained_script
   :caption: examples/training_scripts/constrained_hpo/train_constrained_example.py
   :start-after: # permissions and limitations under the License.


Restrict Scheduler to Tabulated Configurations with Simulator Backend
=====================================================================

.. literalinclude:: ../../examples/launch_lcbench_simulated.py
   :caption: examples/launch_lcbench_simulated.py
   :start-after: # permissions and limitations under the License.

**Requirements**:

* Needs ``benchmarking`` framework, which requires Syne Tune to be installed
  `from source <getting_started.html#installation>`__.
* Needs ``lcbench`` blackbox to be downloaded and preprocessed. This can
  take quite a while when done for the first time
* If `AWS SageMaker is used  <faq.html#how-can-i-run-on-aws-and-sagemaker>`__
  or an S3 bucket is accessible, the blackbox files are uploaded to your S3
  bucket

This example is similar to the one
`above <#launch-hpo-experiment-with-simulator-backend>`__, but here we use
the tabulated LCBench benchmark, whose configuration space is infinite, and
whose objective values have not been evaluated on a grid. With such a
benchmark, we can either use a surrogate to interpolate objective values, or
we can restrict the scheduler to only suggest configurations which have
been observed in the benchmark. This example demonstrates the latter.

Since time is simulated, we can use ``max_wallclock_time=3600`` (one hour),
but the experiment finishes in mere seconds. More details about the simulator
backend is found in
`this tutorial <tutorials/benchmarking/bm_simulator.html>`__.


Tuning Reinforcement Learning
=============================

.. literalinclude:: ../../examples/launch_rl_tuning.py
   :caption: examples/launch_rl_tuning.py
   :start-after: # permissions and limitations under the License.

This launcher script is using the following :ref:`train_cartpole.py <rl_cartpole_script>` training
script:

.. literalinclude:: ../../examples/training_scripts/rl_cartpole/train_cartpole.py
   :name: rl_cartpole_script
   :caption: examples/training_scripts/rl_cartpole/train_cartpole.py
   :start-after: # permissions and limitations under the License.

This training script requires the following dependencies to be
installed:

.. literalinclude:: ../../examples/training_scripts/rl_cartpole/requirements.txt
   :caption: examples/training_scripts/rl_cartpole/requirements.txt


Launch HPO Experiment with SageMaker Backend
============================================

.. literalinclude:: ../../examples/launch_height_sagemaker.py
   :caption: examples/launch_height_sagemaker.py
   :start-after: # permissions and limitations under the License.

**Requirements**:

* `Access to AWS SageMaker <faq.html#how-can-i-run-on-aws-and-sagemaker>`__.
  More details are provided in
  `this tutorial <tutorials/basics/basics_backend.html>`__.
* This example can be sped up by using SageMaker managed warm pools, as in
  `this example <#sagemaker-backend-and-checkpointing>`__.

Makes use of :ref:`train_height.py <train_height_script>`.


SageMaker Backend and Checkpointing
===================================

.. literalinclude:: ../../examples/launch_height_sagemaker_checkpoints.py
   :caption: examples/launch_height_sagemaker_checkpoints.py
   :start-after: # permissions and limitations under the License.

**Requirements**:

* `Access to AWS SageMaker <faq.html#how-can-i-run-on-aws-and-sagemaker>`__.

This launcher script is using the following
:ref:`train_height_checkpoint.py <train_height_checkpoint_script>` training script:

.. literalinclude:: ../../examples/training_scripts/checkpoint_example/train_height_checkpoint.py
   :name: train_height_checkpoint_script
   :caption: examples/training_scripts/checkpoint_example/train_height_checkpoint.py
   :start-after: # permissions and limitations under the License.

Note that :class:`~syne_tune.backend.SageMakerBackend` is configured to use
SageMaker managed warm pools:

* ``keep_alive_period_in_seconds=300`` in the definition of the SageMaker
  estimator
* ``start_jobs_without_delay=False`` when creating :class:`~syne_tune.Tuner`

Managed warm pools reduce both start-up and stop delays substantially, they
are strongly recommended for multi-fidelity HPO with the SageMaker backend.
More details are found in
`this tutorial <tutorials/benchmarking/bm_sagemaker.html#using-sagemaker-managed-warm-pools>`__.


Retrieving the Best Checkpoint
==============================

.. literalinclude:: ../../examples/launch_checkpoint_example.py
   :caption: examples/launch_checkpoint_example.py
   :start-after: # permissions and limitations under the License.

This launcher script is using the following
:ref:`xgboost_checkpoint.py <xgboost_checkpoint.py>` training script:

.. literalinclude:: ../../examples/training_scripts/xgboost/xgboost_checkpoint.py
   :name: xgboost_checkpoint.py
   :caption: examples/training_scripts/xgboost/xgboost_checkpoint.py
   :start-after: # permissions and limitations under the License.


Launch with SageMaker Backend and Custom Docker Image
=====================================================

.. literalinclude:: ../../examples/launch_height_sagemaker_custom_image.py
   :caption: examples/launch_height_sagemaker_custom_image.py
   :start-after: # permissions and limitations under the License.

**Requirements**:

* `Access to AWS SageMaker <faq.html#how-can-i-run-on-aws-and-sagemaker>`__.
* This example is incomplete. If your training script has dependencies which
  you would to provide as a Docker image, you need to upload it to ECR,
  after which you can refer to it with ``image_uri``.

Makes use of :ref:`train_height.py <train_height_script>`.


Launch Experiments Remotely on SageMaker
========================================

.. literalinclude:: ../../examples/launch_height_sagemaker_remotely.py
   :caption: examples/launch_height_sagemaker_remotely.py
   :start-after: # permissions and limitations under the License.

**Requirements**:

* `Access to AWS SageMaker <faq.html#how-can-i-run-on-aws-and-sagemaker>`__.

Makes use of :ref:`train_height.py <train_height_script>`.

This launcher script starts the HPO experiment as SageMaker training job,
which allows you to select any instance type you like, while not having
your local machine being blocked.
`This tutorial <tutorials/benchmarking/README.html>`__ explains how to
run many such remote experiments in parallel, so to speed up comparisons
between alternatives.


Launch HPO Experiment with Home-Made Scheduler
==============================================

.. literalinclude:: ../../examples/launch_height_standalone_scheduler.py
   :caption: examples/launch_height_standalone_scheduler.py
   :start-after: # permissions and limitations under the License.

Makes use of :ref:`train_height.py <train_height_script>`.

For a more thorough introduction on how to develop new schedulers and
searchers in Syne Tune, consider
`this tutorial <tutorials/developer/README.html>`__.


Launch HPO Experiment on mlp_fashionmnist Benchmark
===================================================

.. literalinclude:: ../../examples/launch_fashionmnist.py
   :caption: examples/launch_fashionmnist.py
   :start-after: # permissions and limitations under the License.

**Requirements**:

* Uses ``benchmarking`` framework, which requires Syne Tune to have been
  installed `from source <getting_started.html#installation>`__.

In this example, we tune one of the built-in benchmark problems, which
is useful in order to compare different HPO methods. More details on
benchmarking is provided in
`this tutorial <tutorials/benchmarking/README.html>`__.


Transfer Tuning on NASBench-201
===============================

.. literalinclude:: ../../examples/launch_nas201_transfer_learning.py
   :caption: examples/launch_nas201_transfer_learning.py
   :start-after: # permissions and limitations under the License.

**Requirements**:

* Needs ``nasbench201`` blackbox to be downloaded and preprocessed. This can
  take quite a while when done for the first time
* If `AWS SageMaker is used  <faq.html#how-can-i-run-on-aws-and-sagemaker>`__
  or an S3 bucket is accessible, the blackbox files are uploaded to your S3
  bucket

In this example, we use the simulator backend with the NASBench-201
blackbox. It serves as a simple demonstration how evaluations from
related tasks can be used to speed up HPO.


Transfer Learning Example
=========================

.. literalinclude:: ../../examples/launch_transfer_learning_example.py
   :caption: examples/launch_transfer_learning_example.py
   :start-after: # permissions and limitations under the License.

**Requirements**:

* Needs ``matplotlib`` to be installed if the plotting flag is given:
  ``pip install matplotlib``. If you installed Syne Tune with ``visual`` or
  ``extra``, this dependence is included.

An example of how to use evaluations collected in Syne Tune to run a transfer
learning scheduler. Makes use of :ref:`train_height.py <train_height_script>`.
Used in the
`transfer learning tutorial <tutorials/transfer_learning/transfer_learning.html>`__.
To plot the figures, run as
`python launch_transfer_learning_example.py --generate_plots`.


Plot Results of Tuning Experiment
=================================

.. literalinclude:: ../../examples/launch_plot_results.py
   :caption: examples/launch_plot_results.py
   :start-after: # permissions and limitations under the License.

**Requirements**:

* Needs ``matplotlib`` to be installed:
  ``pip install matplotlib``. If you installed Syne Tune with ``visual`` or
  ``extra``, this dependence is included.

Makes use of :ref:`train_height.py <train_height_script>`.


Customize Results Written during an Experiment
==============================================

.. literalinclude:: ../../examples/launch_height_extra_results.py
   :caption: examples/launch_height_extra_results.py
   :start-after: # permissions and limitations under the License.

Makes use of :ref:`train_height.py <train_height_script>`.

An example for how to append extra results to those written by default to
``results.csv.zip``. This is done by customizing the
:class:`~syne_tune.results_callback.StoreResultsCallback`.


Pass Configuration as JSON File to Training Script
==================================================

.. literalinclude:: ../../examples/launch_height_config_json.py
   :caption: examples/launch_height_config_json.py
   :start-after: # permissions and limitations under the License.

**Requirements**:

* If ``use_sagemaker_backend = True``, needs
  `access to AWS SageMaker <faq.html#how-can-i-run-on-aws-and-sagemaker>`__.

Makes use of the following
:ref:`train_height_config_json.py <train_height_config_json_script>` training
script:

.. literalinclude:: ../../examples/training_scripts/height_example/train_height_config_json.py
   :name: train_height_config_json_script
   :caption: examples/training_scripts/height_example/train_height_config_json.py
   :start-after: # permissions and limitations under the License.


Launch HPO Experiment with Ray Tune Scheduler
=============================================

.. literalinclude:: ../../examples/launch_height_ray.py
   :caption: examples/launch_height_ray.py
   :start-after: # permissions and limitations under the License.

Makes use of :ref:`train_height.py <train_height_script>`.


Stand-Alone Bayesian Optimization
=================================

.. literalinclude:: ../../examples/launch_standalone_bayesian_optimization.py
   :caption: examples/launch_standalone_bayesian_optimization.py
   :start-after: # permissions and limitations under the License.

Syne Tune combines a scheduler (HPO algorithm) with a backend to provide a
complete HPO solution. If you already have a system in place for job
scheduling and managing the state of the tuning problem, you may want to
call the scheduler on its own. This example demonstrates how to do this
for Gaussian process based Bayesian optimization.


Ask Tell Interface
==================

.. literalinclude:: ../../examples/launch_ask_tell_scheduler.py
   :name: launch_ask_tell_scheduler_script
   :caption: examples/launch_ask_tell_scheduler.py
   :start-after: # permissions and limitations under the License.

This is an example on how to use syne-tune in the ask-tell mode.
In this setup the tuning loop and experiments are disentangled. The AskTell Scheduler suggests new configurations
and the users themselves perform experiments to test the performance of each configuration.
Once done, user feeds the result into the Scheduler which uses the data to suggest better configurations.

In some cases, experiments needed for function evaluations can be very complex and require extra orchestration
(example vary from setting up jobs on non-aws clusters to running physical lab experiments) in which case this
interface provides all the necessary flexibility.


Ask Tell interface for Hyperband
================================

.. literalinclude:: ../../examples/launch_ask_tell_scheduler_hyperband.py
   :caption: examples/launch_ask_tell_scheduler_hyperband.py
   :start-after: # permissions and limitations under the License.

This is an extension of
:ref:`launch_ask_tell_scheduler.py <launch_ask_tell_scheduler_script>` to run
multi-fidelity methods such as Hyperband.
