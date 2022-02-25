# Syne Tune: How to Choose a Configuration Space

One important step in applying hyperparameter optimization to your tuning
problem is to define a configuration space (or search space). Doing this
optimally for any given problem is more of an art than a science, but in
this tutorial you will learn about the basics and some gotchas.


### Introduction

Here is an example for a configuration space:

```python
from syne_tune.config_space import randint, uniform, loguniform, \
    choice

config_space = {
    'n_units': randint(4, 1024),
    'dropout': uniform(0, 0.9),
    'learning_rate': loguniform(1e-6, 1),
    'activation': choice(['relu', 'tanh']),
    'epochs': 128,
}
```

Not all entries in `config_space` need to be hyperparameters. For example,
`epochs` is simply a constant passed to the training function. For every
hyperparameter, a domain has to be specified. The domain determines the value
range of the parameter and its internal encoding.

Each hyperparameter is independent of the other entries in `config_space`. In
particular, the domain of a hyperparameter cannot depend on the value of
another. In fact, common actions involving a configuration space, such as
sampling, encoding, or decoding a configuration are done independently on its
hyperparameters.


### Domains

A domain not only defines the value range of a parameter, but also its internal
encoding. The latter is important in order to define what *uniform sampling*
means, a basic component of many HPO algorithms. The following domains are currently supported:

* `uniform(lower, upper)`: Real-valued uniform in `[lower, upper]`
* `loguniform(lower, upper)`: Real-valued log-uniform in
  `[lower, upper]`. More precisely, the value is `exp(x)`, where `x` is drawn
  uniformly in `[log(lower), log(upper)]`
* `randint(lower, upper)`: Integer uniform in `lower, ..., upper`.
  The value range includes both `lower` and `upper` (difference to Python range
  convention)
* `lograndint(lower, upper)`: Integer log-uniform in
  `lower, ..., upper`. More precisely, the value is `int(round(exp(x)))`, where
  `x` is drawn uniformly in `[log(lower - 0.5), log(upper + 0.5)]`
* `choice(categories)`: Uniform from the finite list `categories` of values.
  Entries in `categories` should ideally be of type `str`, but types `int` and
  `float` are also allowed (the latter can lead to errors due to round-off)
* `finrange(lower, upper, size)`: Can be used as finite analogue of `uniform`.
  Uniform from the finite range `lower, ..., upper` of size `size`, where
  entries are equally spaced. For example, `finrange(0.5, 1.5, 3)` means
  `0.5, 1.0, 1.5`, and `finrange(0.1, 1.0, 10)` means `0.1, 0.2, ..., 1.0`.
  We require that `size >= 2`. Note that both `lower` and `upper` are part of
  the value range.
* `logfinrange(lower, upper, size)`: Can be used as finite analogue of
  `loguniform`. Values are `exp(x)`, where `x` is drawn uniformly from the
  finite range `log(lower), ..., log(upper)` of size `size`.  Note that both
  `lower` and `upper` are part of the value range.

By default, the value type for `finrange` and `logfinrange` is `float`. It can
be changed to `int` by the argument `cast_int=True`. For example,
`logfinrange(8, 256, 6, cast_int=True)` results in `8, 16, 32, 64, 128, 256` and
value type `int`, while `logfinrange(8, 256, 6)` results in
`8.0, 16.0, 32.0, 64.0, 128.0, 256.0` and  value type `float`.


### Recommendations

How to choose the domain for a given hyperparameter? Obviously, we want to avoid
illegal values: learning rates should be positive, probabilities lie in `[0, 1]`.
Apart from this, the choice of domain is not always obvious, and different
choices can affect search performance significantly in some cases. Here, we
provide some recommendations.

* **Avoid using `choice` (categorical) for numerical parameters.** Many HPO
  algorithms make very good use of the information that a parameter is
  numerical, therefore has a linear ordering. They cannot do that if you do
  not tell them, and search performance will normally suffer. A good example
  is Bayesian optimization (`scheduler='fifo', searcher='bayesopt'`). Numerical
  parameters are encoded as themselves (the int domain is relaxed to the
  corresponding float interval), allowing the surrogate model (e.g., Gaussian
  process covariance kernel) to exploit ordering and distance in these
  numerical spaces. On the other hand, a categorical parameter with 10
  different values is one-hot encoded to 10(!) dimensions in `[0, 1]`. Now,
  all pairs of distinct values have exactly the same distance in this
  embedding, so that any ordering or distance information is lost. Bayesian
  optimization does not perform well in general in high-dimensional
  embedding spaces.
* **Use infinite ranges.** No competitive HPO algorithm ever enumerates all
  possible configurations and iterates over all of them. There is almost
  certainly no gain in restricting a learning rate to 5 values you just picked
  out of your hat, instead of just using the `loguniform` domain. However,
  there is a lot to be lost. First, you may use `choice`, which may result in
  Bayesian optimization performing poorly. Second, you may just be wrong with
  your initial choice and have to do time-consuming extra steps of refinement.
* **For finite numerical domains, use `finrange` or `logfinrange`.** If you
  insist on a finite range (in some cases, this may be the better choice) for
  a numerical parameter, make use of `finrange` or `logfinrange` instead of
  `choice`, as alternatives to `uniform` and `loguniform` respectively.
* **Use a log transform** for parameters which may vary over several orders of
  magnitude. Examples are learning rates or regularization constants.

