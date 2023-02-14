Installation
============

To install Syne Tune from pip, you can simply do:

.. code-block:: bash

   pip install 'syne-tune[extra]'

For development, you may want to get the latest version from
`github <https://github.com/awslabs/syne-tune>`_:

.. code-block:: bash

   git clone https://github.com/awslabs/syne-tune.git
   cd syne-tune
   python3 -m venv st_venv
   . st_venv/bin/activate
   pip install --upgrade pip
   pip install -e '.[extra]'

This installs everything in a virtual environment ``st_venv``. Remember to activate
this environment before working with Syne Tune. We also recommend building the
virtual environment from scratch now and then, in particular when you pull a new
release, as dependencies may have changed.

See our `change log <https://github.com/awslabs/syne-tune/blob/main/CHANGELOG.md>`_ to
check what has changed in the latest version.

In the examples above, Syne Tune is installed with the union of all its
dependencies, which can be a lot. If you only need specific features, you may
be able to use `partial dependencies <faq.html#what-are-the-different-installations-options-supported>`_.

First Example
=============

To enable tuning, you have to report metrics from a training script so that they
can be communicated later to Syne Tune, this can be accomplished by just
calling :code:`report(epoch=epoch, loss=loss)`, as shown in this example:

.. code-block:: python

   # train_height.py
   import logging
   import time

   from syne_tune import Reporter
   from argparse import ArgumentParser

   if __name__ == '__main__':
       root = logging.getLogger()
       root.setLevel(logging.INFO)
       parser = ArgumentParser()
       parser.add_argument('--epochs', type=int)
       parser.add_argument('--width', type=float)
       parser.add_argument('--height', type=float)
       args, _ = parser.parse_known_args()
       report = Reporter()

       for epoch in range(1, args.epochs + 1):
           time.sleep(0.1)
           dummy_score = 1.0 / (0.1 + args.width * step / 100) + args.height * 0.1
           # Feed the score back to Syne Tune
           report(epoch=epoch, mean_loss=dummy_score)

Once you have annotated your training script in this way, you can launch a
tuning experiment as follows:

.. code-block:: python

   from syne_tune.config_space import randint
   from syne_tune.optimizer.baselines import ASHA
   from syne_tune.backend import LocalBackend
   from syne_tune import Tuner, StoppingCriterion

   # Hyperparameter configuration space
   config_space = {
       "width": randint(1, 20),
       "height": randint(1, 20),
       "epochs": 100,
   }
   # Scheduler (i.e., HPO algorithm)
   scheduler = ASHA(
       config_space,
       metric="mean_loss",
       resource_attr="epoch",
       max_resource_attr="epochs",
       search_options={"debug_log": False},
   )

   tuner = Tuner(
       trial_backend=LocalBackend(entry_point="train_height.py"),
       scheduler=scheduler,
       stop_criterion=StoppingCriterion(max_wallclock_time=15),
       n_workers=4,  # how many trials are evaluated in parallel
   )
   tuner.run()

This example runs `ASHA <tutorials/multifidelity/mf_asha.html>`_ with
``n_workers=4`` asynchronously parallel workers for ``max_wallclock_time=15``
seconds on the local machine it is called on
(:code:`trial_backend=LocalBackend(entry_point="train_height.py")`).

Supported HPO Methods
=====================

The following hyperparameter optimization (HPO) methods are available in Syne Tune:

+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| Method                                                                                | Reference                                                                      | Searcher      | Asynchronous? | Multi-fidelity? | Transfer? |
+=======================================================================================+================================================================================+===============+===============+=================+===========+
| `Grid Search <tutorials/basics/basics_randomsearch.html>`_                            |                                                                                | deterministic | yes           | no              | no        |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `Random Search <tutorials/basics/basics_randomsearch.html>`_                          | `Bergstra, et al. (2011) <https://www.jmlr.org/papers/v13/bergstra12a.html>`_  | random        | yes           | no              | no        |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `Bayesian Optimization <tutorials/basics/basics_bayesopt.html>`_                      | `Snoek, et al. (2012) <https://arxiv.org/abs/1206.2944>`_                      | model-based   | yes           | no              | no        |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.BORE`                                          | `Tiao, et al. (2021) <https://proceedings.mlr.press/v139/tiao21a.html>`_       | model-based   | yes           | no              | no        |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.schedulers.MedianStoppingRule`                           | `Golovin, et al. (2017) <https://dl.acm.org/doi/10.1145/3097983.3098043>`_     | any           | yes           | yes             | no        |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `SyncHyperband <tutorials/multifidelity/mf_syncsh.html>`_                             | `Li, et al. (2018) <https://jmlr.org/papers/v18/16-558.html>`_                 | random        | no            | yes             | no        |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `SyncBOHB <tutorials/multifidelity/mf_sync_model.html#synchronous-bohb>`_             | `Falkner, et al. (2018) <https://arxiv.org/abs/1807.01774>`_                   | model-based   | no            | yes             | no        |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `SyncMOBSTER <tutorials/multifidelity/mf_sync_model.html#synchronous-mobster>`_       | `Klein, et al. (2020) <https://openreview.net/forum?id=a2rFihIU7i>`_           | model-based   | no            | yes             | no        |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `ASHA <tutorials/multifidelity/mf_sync_model.html>`_                                  | `Li, et al. (2019) <https://arxiv.org/abs/1810.05934>`_                        | random        | yes           | yes             | no        |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `BOHB <tutorials/multifidelity/mf_asha.html>`_                                        | `Falkner, et al. (2018) <https://arxiv.org/abs/1807.01774>`_                   | model-based   | yes           | yes             | no        |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `MOBSTER <tutorials/multifidelity/mf_async_model.html#asynchronous-mobster>`_         | `Klein, et al. (2020) <https://openreview.net/forum?id=a2rFihIU7i>`_           | model-based   | yes           | yes             | no        |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `DEHB <tutorials/multifidelity/mf_sync_model.html#differential-evolution-hyperband>`_ | `Awad, et al. (2021) <https://arxiv.org/abs/2105.09821>`_                      | evolutionary  | no            | yes             | no        |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| `HyperTune <tutorials/multifidelity/mf_async_model.html#hyper-tune>`_                 | `Li, et al. (2022) <https://arxiv.org/abs/2201.06834>`_                        | model-based   | yes           | yes             | no        |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.DyHPO`                                         | `Wistuba, et al. (2022) <https://arxiv.org/abs/2202.09774>`_                   | model-based   | yes           | yes             | no        |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.ASHABORE`                                      | `Tiao, et al. (2021) <https://proceedings.mlr.press/v139/tiao21a.html>`_       | model-based   | yes           | yes             | no        |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.PASHA`                                         | `Bohdal, et al. (2022) <https://arxiv.org/abs/2207.06940>`_                    | random        | yes           | yes             | no        |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.REA`                                           | `Real, et al. (2019) <https://arxiv.org/abs/1802.01548>`_                      | evolutionary  | yes           | no              | no        |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.KDE`                                           | `Falkner, et al. (2018) <https://arxiv.org/abs/1807.01774>`_                   | model-based   | yes           | no              | no        |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.schedulers.PopulationBasedTraining`                      | `Jaderberg, et al. (2017) <https://arxiv.org/abs/1711.09846>`_                 | evolutionary  | no            | yes             | no        |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.baselines.ZeroShotTransfer`                              | `Wistuba, et al. (2015) <https://ieeexplore.ieee.org/document/7373431>`_       | deterministic | yes           | no              | yes       |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| ASHA-CTS (:class:`~syne_tune.optimizer.baselines.ASHACTS`)                            | `Salinas, et al. (2021) <https://proceedings.mlr.press/v119/salinas20a.html>`_ | random        | yes           | yes             | yes       |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+
| RUSH (:class:`~syne_tune.optimizer.schedulers.transfer_learning.RUSHScheduler`)       | `Zappella, et al. (2021) <https://arxiv.org/abs/2103.16111>`_                  | random        | yes           | yes             | yes       |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+---------------+---------------+-----------------+-----------+

The searchers fall into four broad categories, **deterministic**, **random**, **evolutionary** and **model-based**. The random searchers sample candidate hyperparameter configurations uniformly at random, while the model-based searchers sample them non-uniformly at random, according to a model (e.g., Gaussian process, density ration estimator, etc.) and an acquisition function. The evolutionary searchers make use of an evolutionary algorithm.

Syne Tune also supports `BoTorch <https://github.com/pytorch/botorch>`_ searchers,
see :class:`~syne_tune.optimizer.baselines.BoTorch`.

Supported multi-objective optimization methods
----------------------------------------------

+-------------------------------------------------------------------------+----------------------------------------------------------------------------+-------------+---------------+-----------------+-----------+
| Method                                                                  | Reference                                                                  | Searcher    | Asynchronous? | Multi-fidelity? | Transfer? |
+=========================================================================+============================================================================+=============+===============+=================+===========+
| :class:`~syne_tune.optimizer.baselines.ConstrainedBayesianOptimization` | `Gardner, et al. (2014) <http://proceedings.mlr.press/v32/gardner14.pdf>`_ | model-based | yes           | no              | no        |
+-------------------------------------------------------------------------+----------------------------------------------------------------------------+-------------+---------------+-----------------+-----------+
| :class:`~syne_tune.optimizer.schedulers.multiobjective.MOASHA`          | `Schmucker, et al. (2021) <https://arxiv.org/abs/2106.12639>`_             | random      | yes           | yes             | no        |
+-------------------------------------------------------------------------+----------------------------------------------------------------------------+-------------+---------------+-----------------+-----------+

HPO methods listed can be used in a multi-objective setting by scalarization
(:class:`~syne_tune.optimizer.schedulers.multiobjective.multiobjective_priority.LinearScalarizationPriority`)
or non-dominated sorting
(:class:`~syne_tune.optimizer.schedulers.multiobjective.multiobjective_priority.NonDominatedPriority`).

Security
========

See `CONTRIBUTING <https://github.com/awslabs/syne-tune/blob/main/CONTRIBUTING.md#security-issue-notifications>`_
for more information.

Citing Syne Tune
================

If you use Syne Tune in a scientific publication, please cite the following paper:

`Syne Tune: A Library for Large Scale Hyperparameter Tuning and Reproducible Research <https://openreview.net/forum?id=BVeGJ-THIg9&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3Dautoml.cc%2FAutoML%2F2022%2FTrack%2FMain%2FAuthors%23your-submissions>`_

.. code-block:: bibtex

   @inproceedings{
       salinas2022syne,
       title = {{Syne Tune}: A Library for Large Scale Hyperparameter Tuning and Reproducible Research},
       author = {David Salinas and Matthias Seeger and Aaron Klein and Valerio Perrone and Martin Wistuba and Cedric Archambeau},
       booktitle = {First Conference on Automated Machine Learning (Main Track)},
       year = {2022},
       url = {https://openreview.net/forum?id=BVeGJ-THIg9}
   }

License
=======

This project is licensed under the Apache-2.0 License.
