Asynchronous Successive Halving
===============================

In this section, we will turn our attention to methods adopting asynchronous
decision-making, which tend to be more efficient than their synchronous
counterparts.

Asynchronous Successive Halving: Early Stopping Variant
-------------------------------------------------------

In synchronous successive halving (SH), decisions on whether to promote a trial
or not can be delayed for a long time. In our example, say we are lucky and
sample an excellent configuration early on, among the 243 initial ones. In
order to promote it to train for 81 epochs, we first need to train 243 trials
for 1 epoch, then 81 for 3 epochs, 27 for 9 epochs, and 9 for 27 epochs. Our
excellent trial will always be among the top :math:`1/3` of others at these
rung levels, but its progress through the rungs is severely delayed.

In `asynchronous successive halving (ASHA) <https://arxiv.org/abs/1810.05934>`__,
the aim is to promote promising configurations as early as possible. There are
two different variants of ASHA, and we will begin with the (arguably) simpler
one. Whenever a worker becomes available, a new configuration is sampled at
random, and a new trial starts training from scratch. Whenever a trial reaches
a rung level, a decision is made *immediately* on whether to stop training or
let it continue. This decision is made based on all data available at the rung
until now. If the trial is among the top :math:`1 / \eta` fraction of
configurations previously registered at this rung, it continues. Otherwise, it
is stopped. As long as a rung has less than :math:`\eta` trials, the default is
to continue.

Different to synchronous SH, there are no fixed rung sizes. Instead, each rung
grows over time. ASHA is free of synchronization points. Promising trials can
be trained for many epochs without having to wait for delayed promotion
decisions. While asynchronous decision-making can be much more efficient at
running good configurations to the end, it runs the risk of making bad
decisions based on too little data.

Our `launcher script <mf_setup.html#the-launcher-script>`__ runs the stopping
variant of ASHA if ``method="ASHA-STOP"``.

Asynchronous Successive Halving: Promotion Variant
--------------------------------------------------

In fact, the algorithm originally proposed as
`ASHA <https://arxiv.org/abs/1810.05934>`__ is slightly different to what has
been detailed above. Instead of starting a trial once and rely on early
stopping, this *promotion variant* is of the *pause-and-resume* type. Namely,
whenever a trial reaches a rung, it is paused there. Whenever a worker becomes
available, all rungs are scanned top to bottom. If a paused trial is found
which lies in the top :math:`1 / \eta` of all rung entries, it is *promoted*:
it may resume and train until the next rung level. If no promotable paused
trial is found, a new trial is started from scratch. Our
`launcher script <mf_setup.html#the-launcher-script>`__ runs the stopping
variant of ASHA if ``method="ASHA-PROM"``.

If these two variants (stopping and promotion) are compared under ideal
conditions, one sometimes does better than the other, and vice versa. However,
they come with different requirements. The promotion variant pauses and resumes
trials, therefore benefits from checkpointing being implemented for the
training code. If this is not the case, the stopping variant may be more
attractive.

On the other hand, the stopping variant requires the backend to frequently stop
workers and bringing them back in order to start a new trial. For some
backends, the turn-around time for this process may be slow, in which case the
promotion type can be more attractive. In this context, it is important to
understand the relevance of passing ``max_resource_attr`` to the scheduler
(and, in our case, also to the
:class:`~syne_tune.blackbox_repository.simulated_tabular_backend.BlackboxRepositoryBackend`).
Recall the discussion `here <mf_setup.html#the-launcher-script>`__. If the
configuration space contains an entry with the maximum resource, whose key is
passed to the scheduler as ``max_resource_attr``, the latter can modify this
value when calling the backend to start or resume a trial. For example, if a
trial is resumed at :math:`r = 3` to train until :math:`r = 9`, the scheduler
passes a configuration to the backend with ``{max_resource_attr: 9}``. This
means that the training code knows how long it has to run, it does not have to
be stopped by the backend.

ASHA can be significantly accelerated by using `PASHA <https://openreview.net/forum?id=syfgJE6nFRW>`__
(Progressive ASHA) that dynamically allocates maximum resources for the tuning
procedure depending on the need. PASHA starts with a small initial amount of
maximum resources and progressively increases them if the ranking of the
configurations in the top two rungs has not stabilized. In practice PASHA
leads to e.g. 3x speedup compared to ASHA, but this can be even higher
for large datasets with millions of examples. A tutorial about PASHA is
`here <../pasha/pasha.html>`__.

Asynchronous Hyperband
----------------------

Finally, ASHA can also be extended to use multiple brackets. Namely, whenever
a new trial is started, its bracket (or, equivalently, its :math:`r_{min}`
value) is sampled randomly from a distribution. In Syne Tune, this distribution
is proportional to the rung sizes in synchronous Hyperband. In our example
with 6 brackets (see details `here <mf_syncsh.html#synchronous-hyperband>`__),
this distribution is :math:`P(r_{min}) = [1:243/415, 3:98/415, 9:41/415,
27:18/415, 81:9/415, 200:6/415]`. Our `launcher script
<mf_setup.html#the-launcher-script>`__ runs asynchronous Hyperband with 6
brackets if ``method="ASHA6-STOP"``.

As also noted in `ASHA <https://arxiv.org/abs/1810.05934>`__, the algorithm
often works best with a single bracket, so that ``brackets=1`` is the default
in Syne Tune. However, we will see further below that model-based variants of
ASHA with multiple brackets can outperform the single-bracket version if the
distribution over :math:`r_{min}` is adaptively chosen.

Finally, Syne Tune implements two variants of ASHA with ``brackets > 1``. In
the default variant, there is only a single system of rungs. For each new
trial, :math:`r_{min}` is sampled to be equal to one of the rung levels, which
means the trial does not have to compete with others at rung levels
:math:`r < r_{min}`. The other variant is activated by passing
``rung_system_per_bracket=True`` to
:class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`. In this case, each
bracket has its own rung system, and trials started in one bracket only have
to compete with others in the same bracket.

Early Removal of Checkpoints
----------------------------

By default, the checkpoints written by all trials are retained on disk (for a
trial, later checkpoints overwrite earlier ones). When checkpoints are large
and the local backend is used, this may result in a lot of disk space getting
occupied, or even the disk filling up. Syne Tune supports checkpoints being
removed once they are not needed anymore, or even speculatively, as is detailed
`here <../../faq.html#checkpoints-are-filling-up-my-disk-what-can-i-do>`__.
