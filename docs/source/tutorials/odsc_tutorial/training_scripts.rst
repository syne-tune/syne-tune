Transformer Trained on WikiText-2
=================================

Here, we train a transformer model on the WikiText-2 dataset. This is a
language modeling problem.

Reporting Once at the End
-------------------------

In the first script, we evaluate the trained model only once, at the end, and
report one metric value back to Syne Tune. This training script works for
random search and Bayesian optimization, but not for ASHA or MOBSTER.

.. literalinclude:: ../../../../benchmarking/nursery/odsc_tutorial/transformer_wikitext2/code/training_script_report_end.py
   :caption: benchmarking/nursery/odsc_tutorial/transformer_wikitext2/code/training_script_report_end.py
   :start-after: # permissions and limitations under the License.

Reporting After Each Epoch
--------------------------

In the second script, we evaluate the model at the end of each epoch and
report results to Syne Tune then. This training script works for ASHA and
MOBSTER as well, as long as they *stop* trials, but not *pause* and *resume*
them.

.. literalinclude:: ../../../../benchmarking/nursery/odsc_tutorial/transformer_wikitext2/code/training_script_no_checkpoints.py
   :caption: benchmarking/nursery/odsc_tutorial/transformer_wikitext2/code/training_script_no_checkpoints.py
   :start-after: # permissions and limitations under the License.

Reporting After Each Epoch With Checkpointing
---------------------------------------------

The final script is like the second, but we also add *checkpointing*. This
training script can be used with all methods implemented in Syne Tune.

.. literalinclude:: ../../../../benchmarking/nursery/odsc_tutorial/transformer_wikitext2/code/training_script.py
   :caption: benchmarking/nursery/odsc_tutorial/transformer_wikitext2/code/training_script.py
   :start-after: # permissions and limitations under the License.
