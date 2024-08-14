Introduction
============

In this section, we define and motivate some basic definitions. As this
tutorial is mostly driven by examples, we will not go into much detail here.

What is Hyperparameter Optimization (HPO)?
------------------------------------------

In *hyperparameter optimization (HPO)*, the goal is to minimize an a priori
unknown function :math:`f(\mathbf{x})` over a *configuration space*
:math:`\mathbf{x}\in\mathcal{X}`. Here, :math:`\mathbf{x}` is a *hyperparameter
configuration*. For example, :math:`f(\mathbf{x})` could be obtained by
training a neural network model on a training dataset, then computing its error
on a disjoint validation dataset. The hyperparameters may configure several
aspects of this setup, for example:

* Optimization parameters: Learning rate, batch size, momentum fraction,
  regularization constant, dropout fraction, choice of stochastic gradient
  descent (SDG) optimizer, warm-up ratio
* Architecture parameters: Number of layers, width of layers, number of
  convolution filters, number of self-attention heads

If HPO ranges over architecture parameters, potentially including the operator
types and connectivity of cells (or layers), it is also referred to as *neural
architecture search (NAS)*.

In general, HPO is a more difficult optimization problem than training for
weights and biases, for a number of reasons:

* Hyperparameters are often discrete (integer or categorical), so smooth
  optimization principles do not apply
* HPO is the outer loop of a nested (or bi-level) optimization problem, where
  the inner loop consists of training for weights and biases. This means that
  an evaluation of :math:`f(\mathbf{x})` can be very expensive (hours or even
  days)
* The nested structure implies further difficulties. Training is
  non-deterministic (random initialization and mini-batch ordering), so
  :math:`f(\mathbf{x})` is really a random function. Even for continuous
  hyperparamters, a gradient of :math:`f(\mathbf{x})` is not tractable to obtain

For these reasons, a considerable amount of technology has so far been applied
to the HPO problem. In the context of this tutorial, two directions are most
relevant:

* Saving compute resources and time by using partial evaluations of
  :math:`f(\mathbf{x})` most of the time. Such evaluations are called *low
  fidelity* or *low resource* below
* Fitting data from :math:`f(\mathbf{x})` (and its lower fidelities) with a
  surrogate probabilistic model. The latter has properties that the real target
  function lacks (fast to evaluate; gradients can be computed), and this can
  efficiently guide the search. The main purpose of a surrogate model is to
  reduce the number of evaluations of :math:`f(\mathbf{x})`, while still finding
  a high quality optimum

Fidelities and Resources
------------------------

In this section, we will introduce concepts of *multi-fidelity hyperparameter
optimization*. Examples will be given further below. The reader may skip this
section and return to it as a glossary.

An evaluation of :math:`f(\mathbf{x})` requires a certain amount of compute
resources and wallclock time. Most of this time is spent in training the model.
In most cases, training resources and time can be broken down into units. For
example:

* Neural networks are trained for a certain number of *epochs* (i.e., sweeps
  over the training set). In this case, *training for one epoch* could be one
  resource unit. This resource unit will be used as running example in this
  tutorial.
* Machine learning models can also be trained on subsets of the training set,
  in order to save resources. We could create a nested system of sets, where for
  simplicity all sizes are integer multiples of the smallest one. In this case,
  *training on the smallest subset size* is one resource unit.

We can decide the amount of resources when evaluating a configuration, giving
rise to observations of :math:`f(\mathbf{x}, r)`, where :math:`r\in\{1, 2, 3,
\dots, r_{max}\}` denotes the resource used (e.g., number of epochs of training).

It is common to define :math:`f(\mathbf{x}, r_{max}) = f(\mathbf{x})`, so that
the original criterion of interest has the largest resource that can be chosen.
In this context, any :math:`f(\mathbf{x}, r)` with :math:`r < r_{max}` is called
a low fidelity criterion w.r.t. :math:`f(\mathbf{x}, r_{max})`. The smaller
:math:`r`, the lower the fidelity. A smaller resource requires less computation
and waiting time, but it also produces a datapoint of less quality when
approximating the target metric. Importantly, all methods discussed here make
the following assumption:

* For every fixed :math:`\mathbf{x}`, running time and compute cost of
  evaluating :math:`f(\mathbf{x}, r)` scales roughly proportional to
  :math:`r`. If this is not the case for the natural resource unit in your
  problem, you need to map :math:`r` to your unit in a non-linear way. Note
  that time may still strongly depend on the configuration :math:`\mathbf{x}`
  itself.

Multi-Fidelity Scheduling
-------------------------

How could an existing HPO technology be extended in order to make use of
multi-fidelity observations :math:`f(\mathbf{x}, r)` at different resources?
There are two basic principles which come to mind:

* A priori decisions: Whenever a decision is required which configuration
  :math:`\mathbf{x}` to evaluate next, the method *also* decides the resource
  :math:`r` to be spent on that evaluation.
* A posteriori decisions: Whenever a new configuration :math:`\mathbf{x}` can
  be run, it is started without a definite amount of resource attached to it.
  After it spent some resources, its low-fidelity observations are compared
  to others who spent the same resource before. Decisions on stopping, or also
  on resuming, trials are taken based on the outcome of such comparisons.

While some work on multi-fidelity Bayesian optimization has chosen the former
option, methods with a posteriori decision-making have been far more successful.
All methods discussed in this tutorial adhere to the a posteriori principle for
decisions which trials to stop or resume from a paused state. In the sequel, we
will use the terminology *scheduling decisions* rather than a posteriori.

How to implement such scheduling decisions? In general, we need to compare a
number of trials with each other on the basis of observations at a certain
resource level :math:`r` (or, more generally, on values up to :math:`r`). In
this tutorial, and in Syne Tune more generally, we use terminology defined in
the `ASHA <https://arxiv.org/abs/1810.05934>`__ publication. A *rung* is a list
of trials :math:`\mathbf{x}_j` and observations :math:`f(\mathbf{x}_j, r)` at a
certain resource level :math:`r`. This resource is also called *rung level*. In
general, a decision on what to do with one or several trials in the rung is
taken by sorting the rung members w.r.t. their metric values. A positive
decision (i.e., continue, or resume) is taken if the trial ranks among the
better ones (above a certain quantile), a negative one (i.e., stop, or keep
paused) is taken otherwise.

More details will be given when we come to real examples below. Just a few
remarks at this point, which will be substantiated with examples:

* Modern successive halving methods innovated over earlier proposals by
  suggesting a geometric spacing of rung levels, and by calibrating the
  thresholds in scheduling decisions according to this spacing. For example,
  the `median stopping rule <https://research.google/pubs/pub46180/>`__
  predates successive halving, but is typically outperformed by ASHA (while MSR
  is implemented in Syne Tune, it is not discussed in this tutorial).
* Scheduling decisions can either be made synchronously or asynchronously. In
  the former case, decisions are batched up for many trials, while in the latter
  case, decisions for each trial are made instantaneously.
* Asynchronous scheduling can either be implemented as *start-and-stop*, or as
  *pause-and-resume*. In the former case, trials are started when workers
  become available, and they may be stopped at rung levels (and just continue
  otherwise). In pause-and-resume scheduling, any trial is always run until the
  next rung level and paused there. When a worker becomes available, it may be
  used to resume any of the paused trials, in case they compare well against
  peers at the same rung. These modalities place `different requirements
  <mf_asha.html#asynchronous-successive-halving-promotion-variant>`__ on the
  training script and the execution backend.
