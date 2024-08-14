Extending Synchronous Hyperband
===============================

In the previous section, we gave an example of how to extend asynchronous
Hyperband with a new searcher. Syne Tune also provides a scheduler template
for *synchronous Hyperband*. In this section, we will walk through an example
of how to extend this template.

Our example here is somewhat more advanced than the one given for asynchronous
Hyperband. In fact, we will walk through the implementation of
`Differential Evolution Hyperband (DEHB) <https://arxiv.org/abs/2105.09821>`__
in Syne Tune. Readers who are not interested in how to extend synchronous
Hyperband, may skip this section without loss.

Synchronous Hyperband
---------------------

The differences between synchronous and asynchronous successive halving and
Hyperband are detailed in
`this tutorial <../multifidelity/mf_asha.html#asynchronous-successive-halving-early-stopping-variant>`__.
In a nutshell, synchronous Hyperband uses rung levels of a priori fixed size,
and decisions on which trials to promote to the next level are only done when
all slots in the current rung are filled. In other words, *promotion decisions*
are synchronized, while the execution of parallel jobs still happens
asynchronously. This requirement poses slight additional challenges for an
implementation, over what is said in
`published work <https://jmlr.org/papers/v18/16-558.html>`__. We start with an
overview of
:class:`~syne_tune.optimizer.schedulers.synchronous.SynchronousHyperbandScheduler`.
Concepts such as resource, rung, bracket, grace period :math:`r_{min}`,
reduction factor :math:`\eta` are detailed in
`this tutorial <../multifidelity/README.html>`__.

:class:`~syne_tune.optimizer.schedulers.synchronous.hyperband_bracket.SynchronousHyperbandBracket`
represents a bracket, consisting of a list of rungs, where each rung is
defined by ``(rung_size, level)``, ``rung_size`` is the number of slots,
``level`` the resource level. Any system of rungs is admissible, as long
as ``rung_size`` is strictly decreasing and ``level`` is strictly
increasing.

* Any active bracket (i.e., supporting running trials) has a
  ``self.current_rung``, where not all slots are occupied.
* A slot in the current rung can be *occupied*, *pending*, or *free*. A slot
  is free if it has not been associated with a trial yet. It is pending if it
  is associated with a trial, but the latter has not returned a metric value
  yet. It is occupied if it contains a metric value. A rung is worked on
  by turning free slots to pending by associating them with a trial, and
  turning pending slots to occupied when their trials return values.
* ``next_free_slot``: Returns ``SlotInRung`` information about the next
  free slot, or ``None`` if all slots are occupied or pending. This method
  is called as part of ``suggest``.
* ``on_result``: This method is called as part of ``on_trial_result``, when a
  trial reports the result a pending slot is waiting for. The corresponding
  slot becomes occupied. If this action renders the rung complete (i.e., all
  slots are occupied), then ``_promote_trials_at_rung_complete`` is called.
  This method increases ``self.current_rung`` and populates the ``trial_id``
  fields by the top performers of the rung just completed. All slots in the new
  rung are free. Note that the ``trial_id`` fields of the first rung are
  assigned to ``None`` at the beginning, they are set by the caller (using
  new ``trial_id`` values provided by the backend).

:class:`~syne_tune.optimizer.schedulers.synchronous.hyperband_bracket_manager.SynchronousHyperbandBracketManager`
maintains all brackets during an experiment. It is configured by a list
of brackets, where each bracket has one less rungs than its predecessor.
The Hyperband algorithm cycles through this ``RungSystemsPerBracket`` in
a round robin fashion. The bracket manager relays ``next_job`` and
``on_result`` calls to the correct ``SynchronousHyperbandBracket``. The
first bracket which is not yet complete, is the *primary bracket*.

* ``next_job``: The preferred bracket to take the job (via ``next_free_slot``)
  is the primary one. However, a bracket may not be able to take the job,
  because its current rung has no free slots (i.e., they are all occupied or
  pending). In this case, the manager scans successive brackets. If no existing
  bracket can take the job, a new bracket is created.

Given these classes,
:class:`~syne_tune.optimizer.schedulers.synchronous.SynchronousHyperbandScheduler`
is straightforward. It is a pause-and-resume scheduler, and it implements the API
:class:`~syne_tune.optimizer.schedulers.multi_fidelity.MultiFidelitySchedulerMixin`,
so that any searchers supporting multi-fidelity schedulers can be used. More
precisely, ``SynchronousHyperbandScheduler`` inherits from
:class:`~syne_tune.optimizer.schedulers.synchronous.hyperband.SynchronousHyperbandCommon`,
which derives from
:class:`~syne_tune.optimizer.schedulers.scheduler_searcher.TrialSchedulerWithSearcher` and
:class:`~syne_tune.optimizer.schedulers.multi_fidelity.MultiFidelitySchedulerMixin`
and collects some code used during construction.

* ``_suggest`` polls ``self.bracket_manager.next_job()``. If the ``SlotInRung``
  returned has ``trial_id`` assigned, it corresponds to a trial to be
  promoted, so the decision is
  :meth:`~syne_tune.optimizer.scheduler.TrialSuggestion.resume_suggestion`
  Otherwise, the scheduler decides for
  :meth:`~syne_tune.optimizer.scheduler.TrialSuggestion.start_suggestion`
  with a new ``trial_id``, which also updates the ``SlotInRung.trial_id`` field.
  In any case, the scheduler maintains the curently pending slots in
  ``self._trial_to_pending_slot``.
* ``on_trial_result`` relays information back via
  ``self.bracket_manager.on_result((bracket_id, slot_in_rung))``, as long
  as ``trial_id`` appears in ``self._trial_to_pending_slot`` and has reached
  its required rung level.

Differential Evolution Hyperband
--------------------------------

We will now have a closer look at the implementation of
`DEHB <https://arxiv.org/abs/2105.09821>`__ in Syne Tune, which is a
recent extension of synchronous Hyperband, where configurations of
trials are chosen by evolutionary computations (mutation, cross-over,
selection). This example is more advanced than the
`one above <extend_async_hb.html>`__, in that we need to do more than
furnishing
:class:`~syne_tune.optimizer.schedulers.synchronous.SynchronousHyperbandScheduler`
with a new searcher. The only time when a searcher suggests configurations is
at the very start, when the first rung of the first bracket is filled. All
further configurations are obtained by evolutionary means.

The main difference between DEHB and synchronous Hyperband is how
configurations to be evaluated in a rung are chosen, based on trials in
the rung above and in earlier brackets. In synchronous Hyperband, we
simply promote the best performing trials from the rung above. In
particular, the configurations do not change, and trials paused in the
rung above are resumed. In DEHB, this promotion process is more
complicated, and importantly, it leads to new trials with different
configurations. This means that trials are not resumed in DEHB.
Moreover, each configuration attached to a trial is represented by an
encoded vector with values in :math:`[0, 1]`, where the mapping from
vectors to configurations is not invertible if the configuration space
contains discrete parameters. Much the same is done in Gaussian process
based `Bayesian optimization <../basics/basics_bayesopt.html>`__.

The very first bracket of DEHB is processed in the same way as in
synchronous Hyperband, so assume the current bracket is not the first.
This is how the configuration vector for a free slot in a rung is
chosen:

* Identify a mutation candidate set. If there is a rung above,
  this set contains the best performing trials from there, namely those
  that would be promoted in synchronous Hyperband. If there is no rung
  above, the set is the rung with same level from the previous bracket.
  Now, if this set contains less than 3 entries, we add configurations
  from earlier trials at the same rung level (the *global parent pool*).
  This mutation candidate set is the same for all choices in the same
  rung.
* Draw 3 configurations at random, without replacement, from the
  mutation candidate set and create a *mutant* as a linear combination of
  them.
* Identify the *target* configuration from the same slot and rung
  level in the previous bracket. The candidate for the slot is obtained by
  *cross-over* between mutant and target, in that each entry of the vector
  is picked randomly from that position in one of the two. An evaluation
  is started for this candidate configuration.
* Finally, there is selection. Once the slot is to be occupied, we compare
  metric values between target and candidate, and the better one gets assigned
  to the slot.

While this sounds quite foreign to what we saw
`above <extend_sync_hb.html#synchronous-hyperband>`__, we can make
progress by associating each candidate vector arising from mutation and
cross-over with a new ``trial_id``. After all, in order to determine the
winner between candidate and target, we have to evaluate the former.
Once this is done, we can map mutation and cross-over to ``suggest``,
and selection to ``on_trial_report``. It becomes clear that we can use
most of the infrastructure for synchronous Hyperband without change.

:class:`~syne_tune.optimizer.schedulers.synchronous.dehb_bracket.DifferentialEvolutionHyperbandBracket`
has only minor differences to ``SynchronousHyperbandBracket``. First,
``_promote_trials_at_rung_complete`` does nothing, because promotion
(i.e., determining the trials for a rung from the one above) is a more
complex process now. In particular, the ``trial_id`` fields of free
slots in the current rung are ``None`` until they become occupied.
Second, ``top_list_for_previous_rung`` returns the top performing trials
of the rung above the current one. This information is needed in order
to create the mutation candidate set. All other methods remain the same.
We still need to identify the next free slot (at the time of mutation
and cross-over), and need to write information back when a slot gets
occupied.

At this point, it is important to acknowledge some difficulties arising
from asynchronous job execution. Namely, mutation and cross-over require
the configurations for the mutation candidate set and target to have
been determined before, and selection needs the metric value for the
target. If this type of information is not present when we need it, we
are not allowed to wait.

* If the current rung is not the first in the bracket, we know that all slots
  in the rung above are occupied. After all, DEHB is still a synchronous HPO
  method.
* The rung from where to choose the target can be problematic, as it may not
  have been decided upon completely when mutation starts for the current rung.
  In this case, our implementation cycles back through the brackets until an
  assigned slot (i.e., not free) is found in the right place.
* For this reason, it is possible in principle that the target ``trial_id``
  changes between cross-over and selection. Also, in rare cases, the target may
  not have a metric at selection time. In this case, the candidate wins.

:class:`~syne_tune.optimizer.schedulers.synchronous.dehb_bracket_manager.DifferentialEvolutionHyperbandBracketManager`
is very similar to ``SynchronousHyperbandBracketManager``. Differences include:

* The system of brackets is more rigid in DEHB, in that subsequent brackets are
  determined by the first one. In particular, later brackets have less total
  budget, because rung sizes are inherited from the first bracket.
* ``top_of_previous_rung`` helps choosing the mutation candidate set. Its
  return values are cached.
* ``trial_id_from_parent_slot`` selects the ``trial_id`` for the target for
  cross-over and selection.

:class:`~syne_tune.optimizer.schedulers.synchronous.DifferentialEvolutionHyperbandScheduler`
implements the DEHB scheduler. Just like ``SynchronousHyperbandScheduler``, it
inherits from
:class:`~syne_tune.optimizer.schedulers.synchronous.hyperband.SynchronousHyperbandCommon`,
which contains common code used by both of them.

* On top of ``SynchronousHyperbandScheduler``, it also maps ``trial_id`` to
  encoded configuration in ``self._trial_info``, and ``self._global_parent_pool``
  maintains all completed trials at each rung level.
* ``_suggest``: We start by determining a free slot, then a configuration vector
  for the new trial, typically by mutation and cross-over. One difficulty is that
  this could end up suggesting a configuration already proposed before,
  because many encoded vectors map to the same configuration. In this
  case, we retry and may ultimately draw encoded configs at random. Except
  for a special case in the very first bracket, we return with
  :meth:`~syne_tune.optimizer.scheduler.TrialSuggestion.start_suggestion`.
* New encoded configurations are chosen only for the first rung of the first
  bracket. Our implementation allows a searcher to be specified for this choice.
  However, the default is to sample the new vector uniformly at random, see
  ``_encoded_config_from_searcher``. Importantly, this is *different* from
  using ``searcher="random"``. The latter samples a configuration and maps
  it to an encoded vector, a process which has less entropy if discrete
  hyperparameters are present.
* ``on_trial_result`` is similar to what happens in
  ``SynchronousHyperbandScheduler``, except that selection is happening as
  well. If the target wins in the selection, ``ext_slot.trial_id`` is changed
  to the target ``trial_id``. In any case, we return ``SchedulerDecision.STOP``
  because the trial will not have to be resumed later on (except in the very
  first bracket).
