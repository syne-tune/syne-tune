Setting up the Problem
======================

Running Example
---------------

For most of this tutorial, we will be concerned with one running example:
tuning some hyperparameters of a two-layer perceptron on the FashionMNIST
dataset.

.. |image1| image:: https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png
.. |image2| image:: https://upload.wikimedia.org/wikipedia/commons/2/2e/Neural_network.png

+-----------------------------------+-----------------------------------+
| FashionMNIST                      | Two-layer MLP                     |
+===================================+===================================+
| |image1|                          | |image2|                          |
+-----------------------------------+-----------------------------------+

This is not a particularly difficult problem. Due to its limited size, and the
type of model, you can run it on a CPU instance. It is not a toy problem
either. Depending on model size, training for the full number of epochs can
take more than 90 minutes. We will present results obtained by running HPO for
3 hours, using 4 workers. In order to get best possible results with
model-based HPO, you would have to run for longer.

Annotating the Training Script
------------------------------

You will normally start with some code to train a machine learning model,
which comes with a number of free parameters you would like to tune. The goal
is to obtain a trained (and tuned) model with low prediction error on future
data from the same task. One way to do this is to split available data into
disjoint training and validation sets, and to score a *configuration* (i.e.,
an instantiation of all hyperparameters) by first training on the training set,
then computing the error on the validation set. This is what we will do here,
while noting that there are other (more costly) scores we could have used
instead (e.g., cross-validation). Here is an example:

.. code-block:: python

   # traincode_report_end.py
   import os
   import argparse
   import logging

   from syne_tune import Reporter


   def download_data(config):
       path = os.path.join(config["dataset_path"], "FashionMNIST")
       os.makedirs(path, exist_ok=True)
       # Lock protection is needed for backends which run multiple worker
       # processes on the same instance
       lock_path = os.path.join(path, "lock")
       lock = SoftFileLock(lock_path)
       try:
           with lock.acquire(timeout=120, poll_intervall=1):
               data_train = datasets.FashionMNIST(
                   root=path, train=True, download=True, transform=transforms.ToTensor()
               )
       except Timeout:
           print(
               "WARNING: Could not obtain lock for dataset files. Trying anyway...",
               flush=True,
           )
           data_train = datasets.FashionMNIST(
               root=path, train=True, download=True, transform=transforms.ToTensor()
           )
       return data_train


   def split_data(config, data_train):
       # We use 50000 samples for training and 10000 samples for validation
       indices = list(range(data_train.data.shape[0]))
       train_idx, valid_idx = indices[:50000], indices[50000:]
       train_sampler = SubsetRandomSampler(train_idx)
       valid_sampler = SubsetRandomSampler(valid_idx)
       batch_size = config["batch_size"]
       train_loader = torch.utils.data.DataLoader(
           data_train, batch_size=batch_size, sampler=train_sampler, drop_last=True
       )
       valid_loader = torch.utils.data.DataLoader(
           data_train, batch_size=batch_size, sampler=valid_sampler, drop_last=True
       )
       return train_loader, valid_loader


   def model_and_optimizer(config):  # [4]
       n_units_1 = config["n_units_1"]
       n_units_2 = config["n_units_2"]
       dropout_1 = config["dropout_1"]
       dropout_2 = config["dropout_2"]
       learning_rate = config["learning_rate"]
       weight_decay = config["weight_decay"]
       # Define the network architecture
       comp_list = [
           nn.Linear(28 * 28, n_units_1),
           nn.Dropout(p=dropout_1),
           nn.ReLU(),
           nn.Linear(n_units_1, n_units_2),
           nn.Dropout(p=dropout_2),
           nn.ReLU(),
           nn.Linear(n_units_2, 10),
       ]
       model = nn.Sequential(*comp_list)
       optimizer = torch.optim.Adam(
           model.parameters(), lr=learning_rate, weight_decay=weight_decay
       )
       criterion = nn.CrossEntropyLoss()
       return {"model": model, "optimizer": optimizer, "criterion": criterion}


   def train_model(config, state, train_loader):
       model = state["model"]
       optimizer = state["optimizer"]
       criterion = state["criterion"]
       batch_size = config["batch_size"]
       model.train()
       for data, target in train_loader:
           optimizer.zero_grad()
           output = model(data.view(batch_size, -1))
           loss = criterion(output, target)
           loss.backward()
           optimizer.step()


   def validate_model(config, state, valid_loader):
       batch_size = config["batch_size"]
       model = state["model"]
       model.eval()
       correct = 0
       total = 0
       for data, target in valid_loader:
           output = model(data.view(batch_size, -1))
           _, predicted = torch.max(output.data, 1)
           total += target.size(0)
           correct += (predicted == target).sum().item()
       return correct / total  # Validation accuracy


   def objective(config):  # [1]
       # Download data
       data_train = download_data(config)
       # Report results to Syne Tune
       report = Reporter()
       # Split into training and validation set
       train_loader, valid_loader = split_data(config, data_train)
       # Create model and optimizer
       state = model_and_optimizer(config)
       # Training loop
       for epoch in range(1, config["epochs"] + 1):
           train_model(config, state, train_loader)

       # Report validation accuracy to Syne Tune
       # [2]
       accuracy = validate_model(config, state, valid_loader)
       report(accuracy=accuracy)


   if __name__ == "__main__":
       # Benchmark-specific imports are done here, in order to avoid import
       # errors if the dependencies are not installed (such errors should happen
       # only when the code is really called)
       from filelock import SoftFileLock, Timeout
       import torch
       import torch.nn as nn
       from torch.utils.data.sampler import SubsetRandomSampler
       from torchvision import datasets
       from torchvision import transforms

       root = logging.getLogger()
       root.setLevel(logging.INFO)
       # [3]
       parser = argparse.ArgumentParser()
       parser.add_argument("--epochs", type=int, required=True)
       parser.add_argument("--dataset_path", type=str, required=True)
       # Hyperparameters
       parser.add_argument("--n_units_1", type=int, required=True)
       parser.add_argument("--n_units_2", type=int, required=True)
       parser.add_argument("--batch_size", type=int, required=True)
       parser.add_argument("--dropout_1", type=float, required=True)
       parser.add_argument("--dropout_2", type=float, required=True)
       parser.add_argument("--learning_rate", type=float, required=True)
       parser.add_argument("--weight_decay", type=float, required=True)
       args, _ = parser.parse_known_args()
       # Evaluate objective and report results to Syne Tune
       objective(config=vars(args))

This is a typical script to train a neural network, using *PyTorch*:

* [1] ``objective`` is encoding the function we would like to optimize. It
  downloads the data, splits it into training and validation set, and
  constructs the model and optimizer. Next, the model is trained for
  ``config['epochs']`` epochs. An epoch constitutes a partitioning of the
  training set into mini-batches of size ``config['batch_size']``,
  presented to the stochastic gradient descent optimizer in a random
  ordering.
* [2] Finally, once training is done, we compute the accuracy of the
  model on the validation set and report it back to Syne Tune. To this end,
  we create a callback (``report = Reporter()``) and call it once the training
  loop finished, passing the validation accuracy
  (``report(accuracy=accuracy)``).
* [3] Values in ``config`` are parameters of the training script. As is
  customary in SageMaker, these parameters are command line arguments to the
  script. A subset of these parameters are *hyperparameters*, namely the
  parameters we would like to tune. Our example has 7 hyperparameters, 3 of
  type int and 4 of type float. Another notable parameter is
  ``config['epochs']``, the number of epochs to train. This is not a parameter
  to be tuned, even though it plays an important role when we get to *early
  stopping* methods below. If your training problem is iterative in nature, we
  recommend you include the number of iterations (or epochs) among the
  parameters to your script.
* [4] Most hyperparameters determine the model, optimizer or learning rate
  scheduler. In ``model_and_optimizer``, we can see that
  ``config['n_units_1']``, ``config['n_units_2']`` are the number of units in
  first and second hidden layer of a multi-layer perceptron with *ReLU*
  activations and dropout (FashionMNIST inputs are 28-by-28 grey-scale images,
  and there are 10 classes). Also, ``config['learning_rate']`` and
  ``config['weight_decay]`` parameterize the Adam optimizer.

This script differs by a vanilla training script only by two lines, which
create ``reporter`` and call it at the end of training. Namely, we report
the validation accuracy after training as ``report(accuracy=accuracy)``.

.. note::
   By default, the configuration is passed to the training script as command
   line arguments. This precludes passing arguments of complex type, such as
   lists or dictionaries, as there is also a length limit to arguments. In
   order to get around these restrictions, you can also pass
   `arguments via a JSON file <../../faq.html#how-can-i-pass-lists-or-dictionaries-to-the-training-script>`__.

Defining the Configuration Space
--------------------------------

Having defined the objective, we still need to specify the space we would like
to search over. The following code is extracted from the complete launcher
script for our experiments, to be introduced below:

.. code-block:: python

   from syne_tune.config_space import randint, uniform, loguniform

   config_space = {
       'n_units_1': randint(4, 1024),
       'n_units_2': randint(4, 1024),
       'batch_size': randint(8, 128),
       'dropout_1': uniform(0, 0.99),
       'dropout_2': uniform(0, 0.99),
       'learning_rate': loguniform(1e-6, 1),
       'weight_decay': loguniform(1e-8, 1),
   }

The configuration space is a dictionary with key names corresponding to command
line input parameters of our training script. For each parameter you would like
to tune, you need to specify a :class:`~syne_tune.config_space.Domain`, imported
from :mod:`syne_tune.config_space`. A domain consists of a type (float, int,
categorical), a range (inclusive on both ends), and an encoding (linear or
logarithmic). In our example, ``n_units_1``, ``n_units_2``, ``batch_size`` are
int with linear encoding (``randint``), ``dropout_1``, ``dropout_2`` are
float with linear encoding (``uniform``), and ``learning_rate``,
``weight_decay`` are float with logarithmic encoding (``loguniform``).
We also need to specify upper and lower bounds: ``n_units_1`` lies between 4
and 1024, the range includes both boundary values.

Choosing a good configuration space for a given problem may require some
iterations. Parameters like learning rate or regularization constants are often
log-encoded, as best values may vary over several orders of magnitude and may
be close to 0. On the other hand, probabilities are linearly encoded. Search
ranges need to be chosen wide enough not to discount potentially useful values
up front, but setting them overly large risks a long tuning time.

In general, the range definitions are more critical for methods based on random
exploration than for model-based HPO methods. On the other hand, we should
avoid to encode finite-sized numerical ranges as categorical for model-based
HPO, instead using one of the more specialized types in Syne Tune. More details
on choosing the configuration space are provided
`here <../../search_space.html>`__, where you will also learn about more types:
categorical, finite range, and ordinal.

Finally, you can also tune only a subset of the hyperparameters of your
training script, providing fixed (default) values for the remaining
ones. For example, the following configuration space fixes the model
architecture:

.. code-block:: python

   from syne_tune.config_space import randint, uniform, loguniform

   config_space = {
       'n_units_1': 512,
       'n_units_2': 128,
       'batch_size': randint(8, 128),
       'dropout_1': uniform(0, 0.99),
       'dropout_2': uniform(0, 0.99),
       'learning_rate': loguniform(1e-6, 1),
       'weight_decay': loguniform(1e-8, 1),
   }
