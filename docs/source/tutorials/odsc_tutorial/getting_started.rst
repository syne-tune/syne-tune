Getting Started with Hyperparameter Tuning
==========================================

In this section, you will learn what is needed to get hyperparameter tuning up
and running. We will look at an example where a deep learning language model
is trained on natural language text.

What is Hyperparameter Tuning?
------------------------------

When solving a business problem with machine learning, there are parts which can
be automated by spending compute resources, and other parts require human
expert attention and choices to be made. By automating some of the more tedious
parts of the latter, hyperparameter tuning shifts the needle between these cost
factors. Like any other smart tool, it saves you time to concentrate on where your
strengths really lie, and where you can create the most value.

At a high level, hyperparameter tuning finds *configurations* of a system which
optimize a *target metric* (or several ones, as we will see later). We can try
any configuration from a *configuration space*, but each evaluation of the system
has a *cost* and takes time. The main challenge of hyperparameter tuning is to
run as few trials as possible, so that total costs are minimal. Also, if
possible, trials should be run in parallel, so that the total experiment time
is minimal.

In this tutorial, we will mostly
be focussed on making decisions and tuning free parameters in the context of
*training machine learning models on data*, so their predictions can be used as
part of a solution to a business problem. There are many other steps between the
initial need and a deployed solution, such as understanding business requirements,
collecting, cleaning and labeling data, monitoring and maintenance. Some of
these can be addressed with automated tuning as well, others need different
tools.

A common paradigm for decision-making and parameter tuning is to try a number of
different configurations and select the best in the end.

* A *trial* consists of training a model on a part of the data (the training
  data). Here, training is an automated process (for example, stochastic
  gradient descent on weight and biases of a neural network model), *given*
  a configuration (e.g., what learning rate is used, what batch size, etc.).
  Then, the trained model is evaluated on another part of the data (validation
  data, disjoint from training data), giving rise to a quality metric (e.g.,
  validation error, AUC, F1), or even several ones. For small datasets, we can
  also use cross-validation, by repeating training and evaluation on a
  number of different splits, reporting the average of validation metrics.
* This metric value (or values) is the response of the system to a
  configuration. Note that the response is stochastic: if we run again with
  the same configuration, we may get a different value. This is because training
  has random elements (e.g., initial weights are sampled, ordering of training
  data).

Enough high level and definitions, let us dive into an example.

Annotating a Training Script
----------------------------

First, we need a script to execute a trial, by training a model and evaluating it.
Since training models is bread and butter to machine learners, you will have no
problem to come up with one. We start with an example:
`training_script_report_end.py <training_scripts.html#reporting-once-at-the-end>`__.
Ignoring the boilerplate, here are the important parts. First, we define the
hyperparameters which should be optimized over:

.. literalinclude:: ../../../../benchmarking/nursery/odsc_tutorial/transformer_wikitext2/code/training_script_report_end.py
   :caption: transformer_wikitext2/code/training_script_report_end.py -- hyperparameters
   :start-at: from syne_tune import Reporter
   :end-before: DATASET_PATH =

* The keys of ``_config_space`` are the hyperparameters we would like to tune
  (``lr``, ``dropout``, ``batch_size``, ``momentum``, ``clip``). It also defines
  their ranges and datatypes, we come back to this
  `below <#choosing-a-configuration-space>`__.
* ``METRIC_NAME`` is the name of the target metric returned, ``MAX_RESOURCE_ATTR``
  the key name for how many epochs to train.

Next, here is the function which executes a trial:

.. literalinclude:: ../../../../benchmarking/nursery/odsc_tutorial/transformer_wikitext2/code/training_script_report_end.py
   :caption: transformer_wikitext2/code/training_script_report_end.py -- objective
   :start-at: def objective(config):
   :end-at: report(**{METRIC_NAME: val_loss})

* The input ``config`` to ``objective`` is a configuration dictionary, containing
  values for the hyperparameters and other fixed parameters (such as the number
  of epochs to train).
* [1] We start with downloading training and validation data. The training data
  loader ``train_data`` depends on hyperparameter ``config["batch_size"]``.
* [2] Next, we create model and optimizer. This depends on the remaining hyperparameters
  in ``config``.
* [3] We then run ``config[MAX_RESOURCE_ATTR]`` epochs of training.
* [4] Finally, we compute the error on the validation data and report it back to
  Syne Tune. The latter is done by creating ``report`` of type ``Reporter`` and
  calling it with a dictionary, using ``METRIC_NAME`` as key.

Finally, the script needs some command line arguments:

.. literalinclude:: ../../../../benchmarking/nursery/odsc_tutorial/transformer_wikitext2/code/training_script_report_end.py
   :caption: transformer_wikitext2/code/training_script_report_end.py -- command line arguments
   :start-at: parser = argparse.ArgumentParser(

* We use an argument parser ``parser``. Hyperparameters can be added by
  ``add_to_argparse(parser, _config_space)``, given the configuration space is
  defined in this script, or otherwise you can do this manually. We also need
  some more inputs, which are not hyperparameters, for example
  ``MAX_RESOURCE_ATTR``.

You can also provide the input to a training script
`as JSON file <../../faq.html#how-can-i-pass-lists-or-dictionaries-to-the-training-script>`__.

Compared to a vanilla training script, we only added two lines, creating
``report`` and calling it for reporting the validation error at the end.

Choosing a Configuration Space
------------------------------

Apart from annotating a training script, making hyperparameters explicit as
inputs, you also need to define a configuration space. In our example, we
add this definition to the script, but you can also keep it separate and
use the same training script with different configuration spaces:

.. literalinclude:: ../../../../benchmarking/nursery/odsc_tutorial/transformer_wikitext2/code/training_script_report_end.py
   :caption: transformer_wikitext2/code/training_script_report_end.py -- configuration space
   :start-at: _config_space = {
   :end-before: DATASET_PATH =

* Each hyperparameters gets assigned a data type and a range. In this example,
  ``batch_size`` is an integer, while ``lr``, ``dropout``, ``momentum``, ``clip``
  are floats. ``lr`` is encoded in log scale.

Syne Tune provides a range of data types. Choosing them well requires a bit of
attention, guidelines are given `here <../../search_space.html>`__.

Specifying Default Values
-------------------------

Once you have annotated your training script and chosen a configuration space,
you have specified all the input Syne Tune needs. You can now specify the
details about your tuning experiment in code, as discussed
`here <../basics/basics_randomsearch.html#launcher-script-for-random-search>`__.
However, Syne Tune provides some tooling in :mod:`syne_tune.experiments` which makes the
life of most users easier, and we will use this tooling in the rest of the
tutorial. To this end, we need to define some defaults about how experiments
are to be run (most of these can be overwritten by command line arguments):

.. literalinclude:: ../../../../benchmarking/nursery/odsc_tutorial/transformer_wikitext2/code/transformer_wikitext2_definition.py
   :caption: transformer_wikitext2/code/transformer_wikitext2_definition.py
   :start-after: # permissions and limitations under the License.

All you need to do is to provide a function (``transformer_wikitext2_benchmark`` here)
which returns an instance of
:class:`~syne_tune.experiments.benchmark_definitions.common.RealBenchmarkDefinition`.
The most important fields are:

* ``script``: Filename of training script.
* ``config_space``: The configuration space to be used by default. This consists
  of two parts. First, the hyperparameters from ``_config``, already discussed
  `above <#choosing-a-configuration-space>`__. Second, ``fixed_parameters`` are
  passed to each trial as they are. In particular, we would like to train for
  40 epochs, so pass ``{MAX_RESOURCE_ATTR: 40}``.
* ``metric``, ``max_resource_attr``, ``resource_attr``: Names of inputs to and
  metrics reported from the training script. If ``mode == "max"``, the target
  metric ``metric`` is maximized, if ``mode == "min"``, it is minimized.
* ``max_wallclock_time``: Wallclock time the experiment is going to run (5 hours
  in our example).
* ``n_workers``: Maximum number of trials which run in parallel (4 in our
  example). The achievable degree of parallelism may be lower, depending on
  which execution backend is used and which hardware instance we run on.

Also, note the role of ``**kwargs`` in the function signature, which allows
to overwrite any of the default values (e.g., for ``max_wallclock_time``,
``n_workers``, or ``instance_type``) with command line arguments.

.. note::
   In the Syne Tune experimentation framework, a tuning problem (i.e., training and
   evaluation script together with defaults) is called a *benchmark*.
   This terminology is used even if the goal of experimentation is not benchmarking
   (i.e., comparing different HPO methods), as is the case in this tutorial here.
