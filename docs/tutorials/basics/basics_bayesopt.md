# Basics of Syne Tune: Bayesian Optimization


[[Previous Section]](basics_randomsearch.md)


## Sequential Model-Based Search

With limited parallel computing resources, experiments are sequential processes,
where trials are started and report results in some ordering. This means that
when deciding on which configuration to explore with any given trial, we can
make use of all metric results reported by earlier trials, given that they
already finished. In the simplest case, with a single worker, a new trial can
start only once all earlier trials finished. We should be able to use this
information in order to make better and better decisions as the experiment
proceeds.

To make this precise, at any given time when a worker comes available, we
need to make a decision which configuration to evaluate with the new trial,
based on (a) which decisions have been made for all earlier trials, and (b)
metric values reported those earlier trials which have already finished. With
more than one worker, the trial set for (a) can be larger than for (b), since
some trials may still be running: their results are *pending*. It is important
to take pending trials into account, since otherwise we risk querying our
objective at redundant configurations. The best way to take information (a)
and (b) into account is by way of a statistical model, leading to
*sequential model-based* decision-making.

What is the challenge for making good "next configuration" decisions? Say
we have already evaluated the objective at a number of configurations,
chosen at random. One idea is to refine the search nearby the configuration
which resulted in the best metric value so far, thereby *exploiting* our
knowledge. Even without gradients, such local search can be highly effective.
On the other hand, it risks getting stuck in a local optimum. Another
extreme is random search, where we *explore* the objective all over the
search space. Choosing between these two extremes, at any given point in
time, is known as *explore-exploit trade-off*, and is fundamental to
sequential model-based search.


## What is Bayesian Optimization?

One of the oldest and most widely used instantiations of sequential model-based
search is *Bayesian optimization*. There are a number of great tutorials and
review articles on Bayesian optimization, and we won't repeat them here:
* [Slides by Ryan Adams](https://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/tutorials/tut8_adams_slides.pdf)
* [Review by Peter Frazier](https://arxiv.org/abs/1807.02811)
* [Video by Peter Frazier](https://www.youtube.com/watch?v=c4KKvyWW_Xk)
* [Video by Nando de Freitas](https://www.youtube.com/watch?v=vz3D36VXefI)
* [Video by Matthew Hoffman](https://www.youtube.com/watch?v=C5nqEHpdyoE)

Most instances of Bayesian optimization work by modelling the objective as
function `f(x)`, where `x` is a configuration from the search space. Given such
a *probabilistic surrogate model*, we can condition it on the observed metric
data (b) in order to obtain a posterior distribution. Finally, we use this
posterior distribution along with additional statistics obtained from the data
(such as for example the best metric value attained so far) in order to compute
a *acquisition function* `a(x)`, an (approximate) maximum of which will be our
suggested configuration. While `a(x)` can itself be difficult to globally
optimize, it is available in closed form and can typically be differentiated
w.r.t. `x`. Moreover, it is important to understand that `a(x)` is not an
approximation to `f(x)`, but instead scores the expected *value* of sampling
the objective at `x`, thereby embodying the explore-exploit trade-off. In
particular, once some `x*` is chosen and included into the set (a), `a(x*)` is
much diminished.

The Bayesian optimization template requires us to make two choices:
* Surrogate model: By far the most common choice is to use Gaussian process
  surrogate models (the tutorials linked above explain the basics of
  Gaussian processes). A Gaussian process is parameterized by a mean and a
  covariance (or kernel) function. In Syne Tune, the default corresponds to
  what is most frequently used in practice: Matern 5/2 kernel with
  automatic relevance determination (ARD). A nice side effect of this choice
  is that the model can learn about the relative relevance of each
  hyperparameter as more metric data is obtained, which allows this form of
  Bayesian optimization to render the "curse of dimensionality" much less
  severe than it is for random search.
* Acquisition function: The default choice in Syne Tune corresponds to the
  most popular choice in practice: expected improvement.


## Launcher Script for Bayesian Optimization

Running Bayesian optimization in Syne Tune has much the same setup as for random
search. Please have a look at [launch_bayesopt.py](scripts/launch_bayesopt.py).
* [1] The only difference to the launcher script for random search is the
  choice of `searcher='bayesopt'` here. In terms of scheduling, we still use
  `FIFOScheduler`.
  However, Bayesian optimization has more knobs to turn, which is why the
  `search_options` argument to `FIFOScheduler` becomes relevant. The full
  list of options is documented in
  [GPFIFOSearcher](../../../syne_tune/optimizer/schedulers/searchers/gp_fifo_searcher.py).
  In our example, we set `num_init_random` to `n_workers + 2`, which is the
  number of initial decisions made by random search, before switching over
  to maximizing the acquisition function.


## Results for Bayesian Optimization

| ![Results for Bayesian Optimization](img/tutorial_rs_bo.png) |
|:-------------------------------------------------------------|
| Results for Bayesian Optimization                            |


Here is how Bayesian optimization performs on our running example, compared to
random search. We used the same conditions (4 workers, 3 hours experiment time,
50 random repetitions).

In this particular setup, Bayesian optimization does not outperform random search
after 3 hours. This is a rather common pattern. Bayesian optimization requires a
certain amount of data in order to learn enough about the objective function (in
particular, about which parameters are most relevant) in order to outperform
random search by targeted exploration and exploitation. If we continued to 4 or 5
hours, we would see a significant difference.


## Recommendations

Here, we collect some additional recommendations. Further details are found
[here](../../schedulers.md#bayesian-optimization-searcher--bayesopt).

### Categorical Hyperparameters

While our running example does not have any, hyperparameters of categorical type
are often used. For example:

```python
from syne_tune.config_space import randint, choice

config_space = {
    'n_units_1': randint(4, 1024),
    # ...
    'activation': choice(['ReLU', 'LeakyReLU', 'Softplus']),
}
```

Here, `activation` could determine the type of activation function. Maybe the
most important recommendation for Bayesian optimization and categorical parameters
is not to use them if you do not have to. If your parameter is numerical, it
admits a linear ordering, which is important information for any optimizer. By
turning it into a categorical parameter, this ordering information is lost. Worse,
in Bayesian optimization, the search space is encoded as multi-dimensional unit cube.
This is a relaxation for `int` values, so one parameter maps to one encoded dimension.
For a categorical parameter, in order to make sure that each value is equidistant
any other, we need to use one-hot encoding, so the encoding dimension is equal to
the number of different values!

In short, while it is tempting to "simplify" our search space by replacing the
`n_units_1` domain `randint(4, 1024)` with
`choice([4, 8, 16, 32, 64, 128, 256, 512, 1024])`, reducing 1021 to 9 distinct
values, this would not make much of a difference for random search, while it would
likely make Bayesian optimization perform worse. Both the acquisition function
and the ARD parameters of our surrogate model would have to be optimized over a
space with 8 more dimensions, and valuable ordering information between
`n_units_1` values would be lost. If you insist on a sparse "regular grid"
value range, you can use `logfinrange(4, 1024, 9)`, which has the same 9
values, but uses a latent `int` representation, which is encoded with a single
number. More information can be found [here](../../search_space.md#recommendations).

### Speeding up Decision-Making

Gaussian process surrogate models have many crucial advantages over other
probabilistic surrogate models typically used in machine learning. But they have
one key disadvantage: inference computations scale cubically in the number
of observations. For most HPO use cases, this is not a problem, since no more
than a few hundred evaluations can be afforded.

If you find yourself in a situation where an experiment can run a thousand
evaluations, there are some `search_options` arguments you can use in order
to speed up Bayesian optimization. The most expensive part of making a
decision consists in refitting the parameters of the GP surrogate model, such
as the ARD parameters of the kernel. While this refitting is essential for
good performance with a small number of observations, it can be thinned out or
even stopped when the dataset gets large. You can use `opt_skip_init_length`,
`opt_skip_period` to this end.


In the [next section](basics_asha.md), we will turn to more powerful scheduling
by way of early  stopping, and learn about asynchronous successive halving and
Hyperband.
