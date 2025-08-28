Installation
============

To install Syne Tune from pip, you can simply do:

.. code-block:: bash

   pip install 'syne-tune[basic]'

For development, you need to install Syne Tune from source:

.. code-block:: bash

   git clone https://github.com/awslabs/syne-tune.git
   cd syne-tune
   python3 -m venv st_venv
   . st_venv/bin/activate
   pip install --upgrade pip
   pip install -e '.[basic,dev]'

This installs Syne Tune in a virtual environment ``st_venv``. Remember to activate
this environment before working with Syne Tune. We also recommend building the
virtual environment from scratch now and then, in particular when you pull a new
release, as dependencies may have changed.

See our `change log <https://github.com/awslabs/syne-tune/blob/main/CHANGELOG.md>`__ to
check what has changed in the latest version.

In the examples above, Syne Tune is installed with the tag ``basic``, which
collects a reasonable number of dependencies. If you want to install all
dependencies, replace ``basic`` with ``extra``. You can further refine this
selection by using
`partial dependencies <faq.html#what-are-the-different-installations-options-supported>`__.

What Is Hyperparameter Optimization?
====================================

Here is an
`introduction to hyperparameter optimization <https://d2l.ai/chapter_hyperparameter-optimization/index.html>`__
in the context of deep learning, which uses Syne Tune for some examples.

First Example
=============

To enable tuning, you have to report metrics from a training script so that they
can be communicated later to Syne Tune, this can be accomplished by just
calling :code:`report(epoch=epoch, loss=loss)`, as shown in this example:

.. literalinclude:: ../../examples/training_scripts/height_example/train_height_simple.py
   :caption: train_height_simple.py
   :start-after: # permissions and limitations under the License.

Once you have annotated your training script in this way, you can launch a
tuning experiment as follows:

.. literalinclude:: ../../examples/launch_height_simple.py
   :caption: launch_height_simple.py
   :start-after: # permissions and limitations under the License.


This example runs `ASHA <tutorials/multifidelity/mf_asha.html>`__ with
``n_workers=4`` asynchronously parallel workers for ``max_wallclock_time=30``
seconds on the local machine it is called on
(:code:`trial_backend=LocalBackend(entry_point=entry_point)`).


Supported HPO Methods
=====================

The following hyperparameter optimization (HPO) methods are available in Syne Tune:

+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| Method                                                                                 | Reference                                                                       | Searcher      | Asynchronous? | Multi-fidelity? | Transfer? |
+========================================================================================+=================================================================================+===============+===============+=================+===========+
| :class:`~syne_tune.optimizer.baselines.RandomSearch`                                   | `Bergstra, et al. (2011) <https://www.jmlr.org/papers/v13/bergstra12a.html>`__  | random        | yes           | no              | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.BoTorch`                                        | `Snoek, et al. (2012) <https://arxiv.org/abs/1206.2944>`__                      | model-based   | yes           | no              | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.BORE`                                           | `Tiao, et al. (2021) <https://proceedings.mlr.press/v139/tiao21a.html>`__       | model-based   | yes           | no              | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.CQR`                                            | `Salinas, et al. (2023) <https://proceedings.mlr.press/v202/salinas23a.pdf>`__  | model-based   | yes           | no              | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.schedulers.MedianStoppingRule`                            | `Golovin, et al. (2017) <https://dl.acm.org/doi/10.1145/3097983.3098043>`__     | any           | yes           | yes             | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.ASHA`                                           | `Li, et al. (2019) <https://arxiv.org/abs/1810.05934>`__                        | random        | yes           | yes             | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.ASHACQR`                                        | `Salinas, et al. (2023) <https://proceedings.mlr.press/v202/salinas23a.pdf>`__  | model-based   | yes           | yes             | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.BOHB`                                           | `Falkner, et al. (2018) <https://arxiv.org/abs/1807.01774>`__                   | model-based   | yes           | yes             | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.ASHABORE`                                       | `Tiao, et al. (2021) <https://proceedings.mlr.press/v139/tiao21a.html>`__       | model-based   | yes           | yes             | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.REA`                                            | `Real, et al. (2019) <https://arxiv.org/abs/1802.01548>`__                      | evolutionary  | yes           | no              | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.KDE`                                            | `Falkner, et al. (2018) <https://arxiv.org/abs/1807.01774>`__                   | model-based   | yes           | no              | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.schedulers.PopulationBasedTraining`                       | `Jaderberg, et al. (2017) <https://arxiv.org/abs/1711.09846>`__                 | evolutionary  | no            | yes             | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.ZeroShotTransfer`                               | `Wistuba, et al. (2015) <https://ieeexplore.ieee.org/document/7373431>`__       | deterministic | yes           | no              | yes       |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.ASHACTS`                                        | `Salinas, et al. (2021) <https://proceedings.mlr.press/v119/salinas20a.html>`__ | random        | yes           | yes             | yes       |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.schedulers.transfer_learning.BoundingBox`                 | `Perrone, et al. (2019) <https://arxiv.org/abs/1909.12552>`__                   | any           | yes           | yes             | yes       |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+

The searchers fall into four broad categories, **deterministic**, **random**, **evolutionary** and **model-based**. The random searchers sample candidate hyperparameter configurations uniformly at random, while the model-based searchers sample them non-uniformly at random, according to a model (e.g., Gaussian process, density ration estimator, etc.) and an acquisition function. The evolutionary searchers make use of an evolutionary algorithm.

We recommend using CQR and ASHA-CQR for single-fidelity and multi-fidelity respectively as it was shown to perform
best in `this paper <https://proceedings.mlr.press/v202/salinas23a.pdf>`__ , we also recommend trying out `BORE` as it
is a close runner-up and may perform better in some cases.


Supported Multi-objective Optimization Methods
----------------------------------------------

+-------------------------------------------------------------------------+-----------------------------------------------------------------------------+-------------+---------------+-----------------+-----------+
| Method                                                                  | Reference                                                                   | Searcher    | Asynchronous? | Multi-fidelity? | Transfer? |
+=========================================================================+=============================================================================+=============+===============+=================+===========+
| :class:`~syne_tune.optimizer.schedulers.multiobjective.MOASHA`          | `Schmucker, et al. (2021) <https://arxiv.org/abs/2106.12639>`__             | random      | yes           | yes             | no        |
+-------------------------------------------------------------------------+-----------------------------------------------------------------------------+-------------+---------------+-----------------+-----------+

Security
========

See `CONTRIBUTING <https://github.com/awslabs/syne-tune/blob/main/CONTRIBUTING.md#security-issue-notifications>`__
for more information.

Citing Syne Tune
================

If you use Syne Tune in a scientific publication, please cite the following paper:

`Syne Tune: A Library for Large Scale Hyperparameter Tuning and Reproducible Research <https://openreview.net/forum?id=BVeGJ-THIg9>`__

.. code-block:: bibtex

   @inproceedings{
       salinas2022syne,
       title = {{Syne Tune}: A Library for Large Scale Hyperparameter Tuning and Reproducible Research},
       author = {David Salinas and Matthias Seeger and Aaron Klein and Valerio Perrone and Martin Wistuba and Cedric Archambeau},
       booktitle = {International Conference on Automated Machine Learning, AutoML 2022},
       year = {2022},
       url = {https://proceedings.mlr.press/v188/salinas22a.html}
   }

License
=======

This project is licensed under the Apache-2.0 License.
