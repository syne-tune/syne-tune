Multi-Fidelity Hyperparameter Tuning
====================================

In our example above, a transformer language model is trained for 40 epochs
before being validated. If a configuration performs poorly, we should find out
earlier, and a lot of time could be saved by stopping poorly performing trials
early. This is what *multi-fidelity HPO* methods are doing. There are different
variants:

* Early stopping ("stopping" type): Trials are not just validated after 40
  epochs, but at the end of every epoch. If a trial is performing worse than many
  others trained for the same number of epochs, it is stopped early.

* Pause and resume ("promotion" type): Trials are generally paused at the end of
  certain epochs, called *rungs*. A paused trial gets *promoted* (i.e., its
  training is resumed) if it does better than a majority of trials who reached
  the same rung.

Syne Tune provides a large number of multi-fidelity HPO methods, more details
are given in
`this tutorial <../multifidelity/README.html>`__. In this section, you learn
what needs to be done to support multi-fidelity hyperparameter tuning.

Annotating a Training Script for Multi-fidelity Tuning
------------------------------------------------------

Clearly, the training script
`training_script_report_end.py <training_scripts.html#reporting-once-at-the-end>`__
won't do for multi-fidelity tuning. These methods need to know validation errors
of models after each epoch of training, while the script above only validates the
model at the end, after 40 epochs of training. A small modification of our
training script,
`training_script_no_checkpoints.py <training_scripts.html#reporting-after-each-epoch>`__,
enables multi-fidelity tuning. The relevant part is this:

.. literalinclude:: ../../../../benchmarking/nursery/odsc_tutorial/transformer_wikitext2/code/training_script_no_checkpoints.py
   :caption: transformer_wikitext2/code/training_script_no_checkpoints.py -- objective
   :start-at: def objective(config):
   :end-at: report(**{RESOURCE_ATTR: epoch, METRIC_NAME: val_loss})

Instead of calling ``report`` only once, at the end, we evaluate the model and
report back at the end of each epoch. We also need to report the number of
epochs done, using ``RESOURCE_ATTR`` as key. The execution backend receives these
reports and relays them to the HPO method, which in turn makes a decision whether
the trial may continue or should be stopped.

Checkpointing
-------------

Instead of *stopping* underperforming trials, some multi-fidelity methods rather
*pause* trials. Any paused trial can be *resumed* in the future if there is evidence
that it outperforms the majority of other trials. If training is very expensive,
pause-and-resume scheduling can work better than early stopping, because any
pause decision can be revisited in the future, while a stopping decision is
final. Moreover, pause-and-resume scheduling does not require trials to be
stopped, which can carry delays in some execution backends.

However, pause-and-resume scheduling needs *checkpointing* in order to work
well. Once a trial is paused, its mutable state is stored in disk. When a trial
gets resumed, this state is loaded from disk, and training can resume exactly
from where it stopped.

Checkpointing needs to be implemented as part of the training script. Fortunately,
Syne Tune provides some tooling to simplify this. Another modification of our
training script,
`training_script.py <training_scripts.html#reporting-after-each-epoch-with-checkpointing>`__,
enables checkpointing. The relevant part is this:

.. literalinclude:: ../../../../benchmarking/nursery/odsc_tutorial/transformer_wikitext2/code/training_script.py
   :caption: transformer_wikitext2/code/training_script.py -- objective
   :start-at: def objective(config):
   :end-at: report(**{RESOURCE_ATTR: epoch, METRIC_NAME: val_loss})

Full details about supporting checkpointing are given in
`this tutorial <../basics/basics_promotion.html#pause-and-resume-checkpointing-of-trials>`__.
In a nutshell:

* [1] Checkpoints have to be written at the end of each epoch, to a path passed
  as command line argument. A checkpoint needs to include the epoch number
  when it was written.
* [2] Before the training loop starts, a checkpoint should be loaded from the
  same place. If one is found, the training loop skips all epochs already
  done. If not, it starts from scratch as usual.
* [3] Syne Tune provides some checkpointing tooling for PyTorch models.

At this point, we have a final version,
`training_script.py <training_scripts.html#reporting-after-each-epoch-with-checkpointing>`__,
of our training script, which can be used with all HPO methods in Syne Tune.
While earlier versions are simpler to implement, we recommend to include
reporting and checkpointing after every epoch in any training script you care
about. When checkpoints become very large, you may run into problems with disk
space, which can be dealt with as described
`here <../../faq.html#checkpoints-are-filling-up-my-disk-what-can-i-do>`__.

.. note::
   The pause-and-resume HPO methods in Syne Tune also work if checkpointing is
   not implemented. However, this means that training for a trial to be resumed
   in fact starts from scratch. The additional overhead makes running these
   methods less attractive. We strongly recommend to implement checkpointing.
