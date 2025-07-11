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

Transfer Tuning on NASBench-201
===============================

.. literalinclude:: ../../examples/launch_nas201_transfer_learning.py
   :caption: examples/launch_nas201_transfer_learning.py
   :start-after: # permissions and limitations under the License.

**Requirements**:

* Syne Tune dependencies ``blackbox-repository`` need to be
  `installed <faq.html#what-are-the-different-installations-options-supported>`__.
* Needs ``nasbench201`` blackbox to be downloaded and preprocessed. This can
  take quite a while when done for the first time
* If `AWS SageMaker is used  <faq.html#how-can-i-run-on-aws-and-sagemaker>`__
  or an S3 bucket is accessible, the blackbox files are uploaded to your S3
  bucket

In this example, we use the simulator backend with the NASBench-201
blackbox. It serves as a simple demonstration how evaluations from
related tasks can be used to speed up HPO.



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


Resume a Tuning Job
===================

.. literalinclude:: ../../examples/launch_resume_tuning.py
   :caption: examples/launch_resume_tuning.py
   :start-after: # permissions and limitations under the License.
