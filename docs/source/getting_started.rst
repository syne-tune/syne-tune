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

Experimentation with Syne Tune
==============================

If you plan to use advanced features of Syne Tune, such as different execution
backends or running experiments remotely, writing launcher scripts like
``examples/launch_height_simple.py`` can become tedious. Syne Tune provides an
advanced experimentation framework, which you can learn about in
`this tutorial <tutorials/experimentation/README.html>`__, or also in
`this one <tutorials/odsc_tutorial/README.html>`__. Examples for the
experimentation framework are given in :mod:`benchmarking.examples` and
:mod:`benchmarking.nursery`.

Supported HPO Methods
=====================

The following hyperparameter optimization (HPO) methods are available in Syne Tune:

+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| Method                                                                                 | Reference                                                                       | Searcher      | Asynchronous? | Multi-fidelity? | Transfer? |
+========================================================================================+=================================================================================+===============+===============+=================+===========+
| `Grid Search <tutorials/basics/basics_randomsearch.html>`__                            |                                                                                 | deterministic | yes           | no              | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `Random Search <tutorials/basics/basics_randomsearch.html>`__                          | `Bergstra, et al. (2011) <https://www.jmlr.org/papers/v13/bergstra12a.html>`__  | random        | yes           | no              | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `Bayesian Optimization <tutorials/basics/basics_bayesopt.html>`__                      | `Snoek, et al. (2012) <https://arxiv.org/abs/1206.2944>`__                      | model-based   | yes           | no              | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.BORE`                                           | `Tiao, et al. (2021) <https://proceedings.mlr.press/v139/tiao21a.html>`__       | model-based   | yes           | no              | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.schedulers.MedianStoppingRule`                            | `Golovin, et al. (2017) <https://dl.acm.org/doi/10.1145/3097983.3098043>`__     | any           | yes           | yes             | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `SyncHyperband <tutorials/multifidelity/mf_syncsh.html>`__                             | `Li, et al. (2018) <https://jmlr.org/papers/v18/16-558.html>`__                 | random        | no            | yes             | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `SyncBOHB <tutorials/multifidelity/mf_sync_model.html#synchronous-bohb>`__             | `Falkner, et al. (2018) <https://arxiv.org/abs/1807.01774>`__                   | model-based   | no            | yes             | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `SyncMOBSTER <tutorials/multifidelity/mf_sync_model.html#synchronous-mobster>`__       | `Klein, et al. (2020) <https://openreview.net/forum?id=a2rFihIU7i>`__           | model-based   | no            | yes             | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `ASHA <tutorials/multifidelity/mf_sync_model.html>`__                                  | `Li, et al. (2019) <https://arxiv.org/abs/1810.05934>`__                        | random        | yes           | yes             | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `BOHB <tutorials/multifidelity/mf_asha.html>`__                                        | `Falkner, et al. (2018) <https://arxiv.org/abs/1807.01774>`__                   | model-based   | yes           | yes             | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `MOBSTER <tutorials/multifidelity/mf_async_model.html#asynchronous-mobster>`__         | `Klein, et al. (2020) <https://openreview.net/forum?id=a2rFihIU7i>`__           | model-based   | yes           | yes             | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `DEHB <tutorials/multifidelity/mf_sync_model.html#differential-evolution-hyperband>`__ | `Awad, et al. (2021) <https://arxiv.org/abs/2105.09821>`__                      | evolutionary  | no            | yes             | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `HyperTune <tutorials/multifidelity/mf_async_model.html#hyper-tune>`__                 | `Li, et al. (2022) <https://arxiv.org/abs/2201.06834>`__                        | model-based   | yes           | yes             | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `DyHPO <tutorials/multifidelity/mf_async_model.html#dyhpo>`__ :sup:`*`                 | `Wistuba, et al. (2022) <https://arxiv.org/abs/2202.09774>`__                   | model-based   | yes           | yes             | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.ASHABORE`                                       | `Tiao, et al. (2021) <https://proceedings.mlr.press/v139/tiao21a.html>`__       | model-based   | yes           | yes             | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `PASHA <tutorials/pasha/pasha.html>`__                                                 | `Bohdal, et al. (2022) <https://arxiv.org/abs/2207.06940>`__                    | random        | yes           | yes             | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.REA`                                            | `Real, et al. (2019) <https://arxiv.org/abs/1802.01548>`__                      | evolutionary  | yes           | no              | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.KDE`                                            | `Falkner, et al. (2018) <https://arxiv.org/abs/1807.01774>`__                   | model-based   | yes           | no              | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.schedulers.PopulationBasedTraining`                       | `Jaderberg, et al. (2017) <https://arxiv.org/abs/1711.09846>`__                 | evolutionary  | no            | yes             | no        |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.ZeroShotTransfer`                               | `Wistuba, et al. (2015) <https://ieeexplore.ieee.org/document/7373431>`__       | deterministic | yes           | no              | yes       |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| ASHA-CTS (:class:`~syne_tune.optimizer.baselines.ASHACTS`)                             | `Salinas, et al. (2021) <https://proceedings.mlr.press/v119/salinas20a.html>`__ | random        | yes           | yes             | yes       |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| RUSH (:class:`~syne_tune.optimizer.schedulers.transfer_learning.RUSHScheduler`)        | `Zappella, et al. (2021) <https://arxiv.org/abs/2103.16111>`__                  | random        | yes           | yes             | yes       |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.schedulers.transfer_learning.BoundingBox`                 | `Perrone, et al. (2019) <https://arxiv.org/abs/1909.12552>`__                   | any           | yes           | yes             | yes       |
+----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+

:sup:`*`: We implement the model-based scheduling logic of DyHPO, but use the
same Gaussian process surrogate models as MOBSTER and HyperTune. The original
source code for the paper is
`here <https://github.com/releaunifreiburg/DyHPO/tree/main>`__.

The searchers fall into four broad categories, **deterministic**, **random**, **evolutionary** and **model-based**. The random searchers sample candidate hyperparameter configurations uniformly at random, while the model-based searchers sample them non-uniformly at random, according to a model (e.g., Gaussian process, density ration estimator, etc.) and an acquisition function. The evolutionary searchers make use of an evolutionary algorithm.

Syne Tune also supports `BoTorch <https://github.com/pytorch/botorch>`__ searchers,
see :class:`~syne_tune.optimizer.baselines.BoTorch`.

Supported Multi-objective Optimization Methods
----------------------------------------------

+-------------------------------------------------------------------------+-----------------------------------------------------------------------------+-------------+---------------+-----------------+-----------+
| Method                                                                  | Reference                                                                   | Searcher    | Asynchronous? | Multi-fidelity? | Transfer? |
+=========================================================================+=============================================================================+=============+===============+=================+===========+
| :class:`~syne_tune.optimizer.baselines.ConstrainedBayesianOptimization` | `Gardner, et al. (2014) <http://proceedings.mlr.press/v32/gardner14.pdf>`__ | model-based | yes           | no              | no        |
+-------------------------------------------------------------------------+-----------------------------------------------------------------------------+-------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.schedulers.multiobjective.MOASHA`          | `Schmucker, et al. (2021) <https://arxiv.org/abs/2106.12639>`__             | random      | yes           | yes             | no        |
+-------------------------------------------------------------------------+-----------------------------------------------------------------------------+-------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.NSGA2`                           | `Deb, et al. (2002) <https://ieeexplore.ieee.org/document/996017>`__        | evolutionary| no            | no              | no        |
+-------------------------------------------------------------------------+-----------------------------------------------------------------------------+-------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.MORandomScalarizationBayesOpt`   | `Peria, et al. (2018) <https://proceedings.mlr.press/v115/paria20a.html>`__ | model-based | yes           | no              | no        |
+-------------------------------------------------------------------------+-----------------------------------------------------------------------------+-------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.MOLinearScalarizationBayesOpt`   |                                                                             | model-based | yes           | no              | no        |
+-------------------------------------------------------------------------+-----------------------------------------------------------------------------+-------------+---------------+-----------------+-----------+

HPO methods listed can be used in a multi-objective setting by scalarization
(:class:`~syne_tune.optimizer.schedulers.multiobjective.multiobjective_priority.LinearScalarizationPriority`)
or non-dominated sorting
(:class:`~syne_tune.optimizer.schedulers.multiobjective.multiobjective_priority.NonDominatedPriority`).

Security
========

See `CONTRIBUTING <https://github.com/awslabs/syne-tune/blob/main/CONTRIBUTING.md#security-issue-notifications>`__
for more information.

Citing Syne Tune
================

If you use Syne Tune in a scientific publication, please cite the following paper:

`Syne Tune: A Library for Large Scale Hyperparameter Tuning and Reproducible Research <https://openreview.net/forum?id=BVeGJ-THIg9&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3Dautoml.cc%2FAutoML%2F2022%2FTrack%2FMain%2FAuthors%23your-submissions>`__

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
