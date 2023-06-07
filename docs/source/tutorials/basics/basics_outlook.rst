Outlook
=======

Further Topics
--------------

We are at the end of this basic tutorial. There are many further topics we did
not touch here. Some are established, but not basic, while others are still
experimental. Here is an incomplete overview:

* **Running many experiments in parallel**: We have stressed the importance
  of running repetitions of experiments, as results carry quite some stochastic
  variation. Also, there are higher-level decisions best done by
  trial-and-error, which can be seen as “outer loop random search”. Syne Tune
  offers facilities to launch many tuning experiments in parallel, as SageMaker
  training jobs. More details are found in
  `this tutorial <../benchmarking/README.html>`__, see also the
  `FAQ <../../faq.html#how-can-i-run-many-experiments-in-parallel>`__.
* **Multi-fidelity Schedulers**: Syne Tune provides many more multi-fidelity
  schedulers than ASHA and MOBSTER. An overview and categorization of supported
  methods is given in `this tutorial <../multifidelity/README.html>`__.
* **Population-based Training**: This is a popular scheduler for tuning
  reinforcement learning, where optimization hyperparameters like learning rate
  can be changed at certain points during the training. An example is at
  ``examples/launch_pbt.py``, see also
  :class:`~syne_tune.optimizer.schedulers.PopulationBasedTraining`. Note that
  `checkpointing <basics_promotion.html#pause-and-resume-checkpointing-of-trials>`__
  is mandatory for PBT.
* **Constrained HPO**: In many applications, more than a single metric play a
  role. With constrained HPO, you can maximize recall subject to a constraint
  on precision; minimize prediction latency subject to a constraint on
  accuracy; or maximize accuracy subject to a constraint on a fairness metric.
  Constrained HPO is a special case of
  `Bayesian Optimization <basics_bayesopt.html>`__, where
  ``searcher='bayesopt_constrained'``, and the name of the constraint metric
  (the constraint is feasible iff this metric is non-positive) must be given
  as ``constraint_attr`` in ``search_options``. More details on constrained HPO
  and methodology adopted in Syne Tune can be found
  `here <https://arxiv.org/abs/1910.07003>`__, see also
  :class:`~syne_tune.optimizer.schedulers.searchers.constrained.ConstrainedGPFIFOSearcher`.
* **Multi-objective HPO**: Another way to approach tuning problems with multiple
  metrics is trying to sample the Pareto frontier, i.e. identifying
  configurations whose performance along one metric cannot be improved without
  degrading performance along another. Syne Tune provides a range of methodology
  in this direction. An example is at ``examples/launch_height_moasha.py``.
  More details on multi-objective HPO and methodology adopted in Syne Tune can
  be found `here <https://arxiv.org/abs/2106.12639>`__, see also
  :class:`~syne_tune.optimizer.schedulers.multiobjective.MOASHA`.
* **Transfer-learning Schedulers**: Syne Tune provides several transfer-learning schedulers. To get started check out `this tutorial <../transfer_learning/transfer_learning.html>`__.
