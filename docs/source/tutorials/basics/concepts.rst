Basics of Syne Tune: Concepts and Terminologies
======================

Syne Tune is a library for large-scale distributed hyperparameter optimization (HPO). Some key terminologies in HPO domain are the followings:
First, a specific settings/values of hyperparameters is referred to as *configuration*.
Second, the *configuration space* is the domain of a configuration i.e. scale and range of each hyperparameter.
And third, the evaluation of the underlying ML task on a given configuration is referred as *trial*.


Syne Tune components
--------------------
Syne Tune consists of four core components: *Tuner,  Backend, Scheduler* and *Benchmarking*.
These components together facilitate selection of new trials (i.e., evaluations of configurations), launching, pausing, resuming or interrupting
them, and retrieving the evaluation results. In what follows, we describe each component:


Tuner
-----
On a high level, *Tuner* orchestrates the overall search for the best configuration. It does so by interacting with *scheduler* and *backend*.
It queries the *Scheduler* for a trial when a worker is free, and pass the trial to the backend
for execution.  

Scheduler
---------
In Syne Tune, HPO algorithms are called *Schedulers*.
Schedulers search for a new, most promising configuration and suggest it as a new trial to *Tuner*.
Some schedulers may decide to resume a paused trial instead of suggesting a new one.
Schedulers are also in charge of stopping running trials. Syne Tune supports many schedulers including multi-fidelity ones and asynchronous ones.

Backend
-------
The Backend module is responsible for starting, stopping, pausing and resuming trials and accessing
results and trial statuses. Syne Tune currently supports four execution backends to facilitate experimentations: local backend, python backend, SageMaker backend, and simulated backend. 
To explain these backends, recall that any HPO job consists of two main scripts: a tuner script where HPO loop runs and next configurations from configuration space is selected, and a training script where the machine learning model (whether it is a neural network or a tree-based model) is trained based on the selected hyperparameter configuration from tuner.
The execution backend runs these scripts; and depending on whether they are executed on one or multiple machines, the execution backend changes.  

Local backend runs the training job locally with respect to where tuner job is running. That is if both tuner script and training script run on the same machine (whether it is your local machine or in cloud) it is local backend. 
Figures below demonstrates local backend, where in the left figure both scripts are executed on local machine,
and in right figure scripts are executed on a SageMaker instance in cloud.


.. |image1| image:: img/local1.png
            :width: 200
.. |image2| image:: img/local2.png
            :width: 530
+----------------------------------------------------------+-------------------------------------------------------------+
| |image1|                                                 | |image2|                                                    |
+==========================================================+=============================================================+
| Local backend on a local machine                         | Local backend when running on a SageMaker instance on cloud |
+----------------------------------------------------------+-------------------------------------------------------------+


Local backend evaluates trials concurrently on a single machine by using subprocesses.
SyneTune support rotating multiple GPUs on the machine, assigning the next trial to the least
busy GPU, e.g. the GPU with the smallest amount of trials currently running. 

Local backend suffers from two shortcomings; first, it limits the number of trials that can run concurrently.
Second, it falls short in training neural network which require many GPUs, or possibly distributed across several nodes.
Python backend is just a wrapper around the local backend, shortly we see an example how the two differ. 

SageMaker backend fills above gap by executing training script in cloud and specifically in an Amazon SageMaker instance while the tuner script runs either from your local machine or from another SageMaker instance in cloud. 
Figure x and y show these settings respectively. 
SageMaker backend schedules one training job per trial. 
In addition, Amazon SageMaker provides pre-build containers of ML frameworks
(e.g., Pytorch, TensorFlow, Scikit-learn, HuggingFace) and enables users of training on cheaper preemptible machines.


.. |image3| image:: img/sm_backend1.png
            :width: 500
.. |image4| image:: img/sm_backend2.png
            :width: 700
+----------------------------------------------------------+-------------------------------------------------------------------------+
|  |image3|                                                | |image4|                                                                |
+==========================================================+=========================================================================+
| SageMaker backend with tuner running from local machine  | SageMaker backend with both tuner and training scripts running on cloud |
+----------------------------------------------------------+-------------------------------------------------------------------------+



In SageMaker backend, each trial is run as a separate SageMaker training job. This is useful for expensive workloads,
where all resources of an instance (or several ones) are used for training. On the other hand, training job start-up overhead is incurred for every trial.


Note that Syne Tune is agnostic to execution backend, 
and users can effortlessly change between backends by modifying input argument ``trial_backend`` in instantiating `Tuner`.
See `launch_randomsearch.py <scripts/launch_randomsearch.py>`__
for an example of local backend where ``entry_point`` is the training script.
See `launch_height_python_backend.py <scripts/launch_height_python_backend.py>`__
for an example of Python backend, where the training script is
just a training function (in this example ``train_height()`` function) located in the tuner script.
See `launch_sagemaker_backend.py <scripts/launch_sagemaker_backend.py>`__ for an example of SageMaker backend, where
a PyTorch container on ``ml.m4.xlarge`` instance is picked to run the training script (i.e.``entry_point``).

[TODO] simulated backend

Benchmarking
------------



