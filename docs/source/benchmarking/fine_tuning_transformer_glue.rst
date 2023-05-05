Code in benchmarking/examples/fine_tuning_transformer_glue
==========================================================

Selecting pre-trained transformer model from Hugging Face zoo and
fine-tuning it to a GLUE task. This is in fact a whole family of benchmarks:

* :code:`f"finetune_transformer_glue_{dataset}"`: Tune number of hyperparameters
  for fixed pre-trained model, selected by ``--model_type``
* :code:`f"finetune_transformer_glue_modsel_{dataset}"`: Tune the same
  hyperparameters and select the best pre-trained model from a list of 9
  choices

Here, ``dataset`` selects the GLUE document classification task (values are
"cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli").

.. literalinclude:: ../../../benchmarking/examples/fine_tuning_transformer_glue/baselines.py
   :caption: benchmarking/examples/fine_tuning_transformer_glue/baselines.py
   :start-after: # permissions and limitations under the License.

.. literalinclude:: ../../../benchmarking/examples/fine_tuning_transformer_glue/hpo_main.py
   :caption: benchmarking/examples/fine_tuning_transformer_glue/hpo_main.py
   :start-after: # permissions and limitations under the License.

.. literalinclude:: ../../../benchmarking/examples/fine_tuning_transformer_glue/launch_remote.py
   :caption: benchmarking/examples/fine_tuning_transformer_glue/launch_remote.py
   :start-after: # permissions and limitations under the License.

.. literalinclude:: ../../../benchmarking/examples/launch_local/requirements-synetune.txt
   :caption: benchmarking/examples/launch_local/requirements-synetune.txt
