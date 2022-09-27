# Multi-Fidelity HPO: Model-based Asynchronous Hyperband

We have seen that asynchronous decision-making tends to outperform synchronous
variants in practice, and model-based extensions of the latter can outperform
random sampling of new configurations. In this section, we discuss combinations
of Bayesian optimization with asynchronous decision-making, leading to the
currently best performing multi-fidelity methods in Syne Tune.

All examples here can either be run in stopping or promotion mode of ASHA. We
will use the promotion mode here (i.e., pause-and-resume scheduling).


## Surrogate Models of Learning Curves

[Recall](mf_syncsh.md#early-stopping-hyperparameter-configurations) that validation
error after $r$ epochs is denoted by $f(\mathbf{x}, r)$, where
$\mathbf{x}$ is the configuration. The function
$r\mapsto f(\mathbf{x}, r)$ is called learning curve. A learning curve
surrogate model predicts $f(\mathbf{x}, r)$ from observed data. A difficult
requirement in the context of multi-fidelity HPO is that observations are much
more abundant at smaller resource levels $r$, while predictions are more valuable
at larger $r$.

In the context of Gaussian process based
[Bayesian optimization](../basics/basics_bayesopt.md), Syne Tune supports a
number of different learning curve surrogate models. The type of model is
selected upon construction of the scheduler:

```python
scheduler = HyperbandScheduler(
    config_space,
    type="promotion",
    searcher="bayesopt",
    search_options=dict(
        model="gp_multitask",
        gp_resource_kernel="exp-decay-sum",
    ),
    metric=benchmark.metric,
    mode=benchmark.mode,
    resource_attr=resource_attr,
    random_seed=random_seed,
    max_resource_attr=max_resource_attr,
)
```

First, `searcher="bayesopt"` is selecting MOBSTER as searcher in asynchronous
Hyperband. Further options configuring the searcher are collected in
`search_options`. The most important options are `model`, selecting the type of
surrogate model, and `gp_resource_kernel` selecting the covariance model in the
case `model="gp_multitask"`.

### Independent Processes at each Rung Level

A simple learning curve surrogate model is obtained by
`search_options["model"] = "gp_independent"`. Here, $f(\mathbf{x}, r)$ at
each rung level $r$ is represented by an independent Gaussian process model.
The models have individual constant mean functions
$\mu_r(\mathbf{x}) = \mu_r$ and covariance functions
$k_r(\mathbf{x}, \mathbf{x}') = v_r k(\mathbf{x}, \mathbf{x}')$, where
$k(\mathbf{x}, \mathbf{x}')$ is a Matern-5/2 ARD kernel without variance
parameter, which is shared between the models, and the $v_r > 0$ are
individual variance parameters. The idea is that while validation errors at
different rung levels may be scaled and shifted, they should still exhibit
similar dependencies on the hyperparameters. The noise variance $\sigma^2$
used in the Gaussian likelihood is the same across all data. However, if
`search_options["separate_noise_variances"] = True`, different noise variances
$\sigma_r^2$ are used for data at different rung levels.

### Multi-Task Gaussian Process Models

A more advanced set of learning curve surrogate models is obtained by
`search_options["model"] = "gp_multitask"` (which is the default for asynchronous
MOBSTER). In this case, a single Gaussian process model represents
$f(\mathbf{x}, r)$ directly, with mean function $\mu(\mathbf{x}, r)$
and covariance function $k((\mathbf{x}, r), (\mathbf{x}', r'))$. The GP model
is selected by `search_options["gp_resource_kernel"]`, currently supported options
are `"exp-decay-sum"`, `"exp-decay-combined"`, `"exp-decay-delta1"`, `"freeze-thaw"`,
`"matern52"`, `"matern52-res-warp"`, `"cross-validation"`. The default choice is
`"exp-decay-sum"`, which is inspired by the exponential decay model proposed
[here](https://arxiv.org/abs/1406.3896). Details about these different models are
given [here](https://openreview.net/forum?id=a2rFihIU7i) and in the source code.

Decision-making is somewhat more expensive with `"gp_multitask"` than with
`"gp_independent"`, because the notorious cubic scaling of GP inference applies
over observations made at all rung levels. However, the extra cost is limited by
the fact that most observations by far are made at the lowest resource level
$r_{min}$ anyway.

### Additive Gaussian Models

Two additional models are selected by `search_options["model"] = "gp_expdecay"`
and `search_options["model"] = "gp_issm"`. The former is the exponential decay
model proposed [here](https://arxiv.org/abs/1406.3896), the latter is a variant
thereof. These additive Gaussian models represent dependencies across $r$ in a
cheaper way than in `"gp_multitask"`, and they can be fit to all observed data,
not just at rung levels. Also, joint sampling is cheap.

However, at this point, additive Gaussian models remain experimental, and they
will not be further discussed here. They can be used with MOBSTER, but not
with Hyper-Tune.


## Asynchronous MOBSTER

[MOBSTER](https://openreview.net/forum?id=a2rFihIU7i) combines ASHA and
asynchronous Hyperband with GP-based Bayesian optimization. A Gaussian process
learning curve surrogate model is fit to the data at all rung levels, and
posterior predictive distributions are used in order to compute acquisition
function values and decide on which configuration to start next. We
distinguish between MOBSTER-JOINT with a GP multi-task model (`"gp_multitask"`)
and MOBSTER-INDEP with an independent GP model (`"gp_independent"`), as
detailed above. The acquisition function is expected improvement (EI) at the
rung level $r_{acq}$ also used by [BOHB](mf_sync_model.md#synchronous-bohb).

A launcher script for (asynchronous) MOBSTER-JOINT is given in
[launch_method.py](scripts/launch_method.py), passing `method="MOBSTER-JOINT"`.
The searcher can be configured with `search_options`, but MOBSTER-JOINT with
the `"exp-decay-sum"` covariance model is the default

As shown [below](mf_comparison.md), MOBSTER can outperform ASHA significantly.
This is achieved by starting many less trials that stop very early (after 1
epoch) due to poor performance. Essentially, MOBSTER rapidly learns some
important properties about the NASBench-201 problem and avoids basic mistakes
which random sampling of configurations runs into at a constant rate. While ASHA
stops such poor trials early, they still take away resources, which MOBSTER can
spend on longer evaluations of more promising configurations. This advantage of
model-based over random sampling based multi-fidelity methods is even more
pronounced when starting and stopping jobs comes with delays. Such delays are
typically present in real world distributed systems, but are absent in our
simulations.

Different to BOHB, MOBSTER takes into account *pending evaluations*,
i.e. trials which have been started but did not return metric values yet. This
is done by integrating out their metric values by Monte Carlo. Namely, we draw
a certain number of joint samples over pending targets and average the
acquisition function over these. In the context of multi-fidelity, if a trial
is running, a pending evaluation is registered for the next recent rung level
it will reach.

Why is the surrogate model in MOBSTER-JOINT fit to the data at rung
levels only? After all, training scripts tend to report validation errors after
each epoch, why not use all this data? Syne Tune allows to do so (for the
`"gp_multitask"` model), by passing `searcher_data="all"` when creating the
`HyperbandScheduler` (another intermediate is `searcher_data="rungs_and_last"`).
However, while this may lead to a more accurate model, it also becomes more
expensive to fit, and does not tend to make a difference, so the default
`searcher_data="rungs"` is recommended.

Finally, we can also combine ASHA with [BOHB](mf_sync_model.md#synchronous-bohb)
decision-making, by choosing `searcher="kde"` in `HyperbandScheduler`. This is
an asynchronous version of BOHB.

### MOBSTER-INDEP

A launcher script for (asynchronous) MOBSTER-INDEP is given in
[launch_method.py](scripts/launch_method.py), passing `method="MOBSTER-INDEP"`.
The independent GPs model is selected by `search_options["model"] = "gp_independent"`.

MOBSTER tends to perform slightly better with a joint multi-task GP model than
with an independent GPs model, justifying the Syne Tune default. In our
experience so far, changing the covariance model in MOBSTER-JOINT has only
marginal impact.

### MOBSTER and Hyperband

Just like [ASHA can be run with multiple brackets](mf_asha.md#asynchronous-hyperband),
so can MOBSTER, simply by selecting `brackets` when creating `HyperbandScheduler`.
In our experience so far, just like with ASHA, MOBSTER tends to work best with
a single bracket.


## Hyper-Tune

[Hyper-Tune](https://arxiv.org/abs/2201.06834) is a model-based extension of ASHA
with some additional features compared to MOBSTER. It can be seen as extending
MOBSTER-INDEP (with the `"gp_independent"` surrogate model) in two ways. First,
it uses an acquisition function based on an ensemble predictive distribution,
while MOBSTER relies on the $r_{acq}$ heuristic from BOHB. Second, if
multiple brackets are used (Hyperband case), Hyper-Tune offers an adaptive
mechanism to sample the bracket for a new trial. Both extensions are based on
a quantification of consistency of data on different rung levels, which is used
to weight rung levels according to their reliability for making decisions
(namely, which configuration $\mathbf{x}$ and bracket $r_{min}$ to
associate with a new trial).

Before diving into details, a launcher script for Hyper-Tune (with one bracket)
is given in
[launch_method.py](scripts/launch_method.py), passing `method="HYPERTUNE-INDEP"`.
The searcher can be configured with `search_options`, but the independent GPs
model `"gp_independent"` is the default

In this example, Hyper-Tune is using a single bracket, so the difference to
MOBSTER-INDEP is due to the ensemble predictive distribution for the
acquisition function.

Syne Tune also implements Hyper-Tune with the GP multi-task surrogate models used in
MOBSTER. In result plots for this tutorial, original Hyper-Tune is called
HYPERTUNE-INDEP, while this latter variant is called HYPERTUNE-JOINT. A launcher
script is given in
[launch_method.py](scripts/launch_method.py), passing `method="HYPERTUNE-JOINT"`.

### Hyper-Tune with Multiple Brackets

Just like ASHA and MOBSTER, Hyper-Tune can also be run with multiple brackets,
simply by using the `brackets` argument of `HyperbandScheduler`. If
`brackets > 1`, Hyper-Tune samples the bracket for a new trial from an adaptive
distribution closely related to the ensemble distribution used for acquisitions.
A launcher script is given in
[launch_method.py](scripts/launch_method.py), passing `method="HYPERTUNE4-INDEP"`.

Recall that both ASHA and MOBSTER tend to work better for one than for multiple
brackets. This may well be due to the fixed, non-adaptive distribution that
brackets are sampled from. Ideally, a method would learn over time whether a low
rung level tends to be reliable in predicting the ordering at higher ones, or
whether it should rather be avoided (and $r_{min}$ should be increased).
This is what the adaptive mechanism in Hyper-Tune tries to do. In our
comparisons, we find that HYPERTUNE-INDEP with multiple brackets can outperform
MOBSTER-JOINT with a single bracket.

### Details

In this section, we provide some details about Hyper-Tune and our implementation.
The Hyper-Tune extensions are based on a quantification of consistency of data on
different rung levels For example, assume that $r < r_{*}$ are two rung
levels, with sufficiently many points at $r_{*}$. If $\mathcal{X}_{*}$
collects trials with data at $r_{*}$, all these have also been observed at
$r$. Sampling $f(\mathcal{X}_{*}, r)$ from the posterior distribution
of the surrogate model, we can compare the *ordering* of these predictions at
$r$ with the ordering of observations at $r_{*}$, using a pair-wise
ranking loss. A large loss value means frequent cross-overs of learning curves
between $r$ and $r_{*}$, and predictions at rung level $r$ are
unreliable when it comes to the ordering of trials $\mathcal{X}_{*}$ at
$r_{*}$.

At any point during the algorithm, denote by $r_{*}$ the largest rung
level with a sufficient number of observations (our implementation
requires 6 points). Assuming that $r_{*} > r_{min}$, we can estimate a
distribution $[\theta_r]$ over rung levels
$\mathcal{R}_{*} = \{r\in\mathcal{R}\, |\, r\le r_{*}\}$ as follows.
We draw $S$ independent samples from the model at these rung levels.
For each sample $s$, we compute loss values $l_{r, s}$ for
$(r, r_{*})$ over all $r\in\mathcal{R}_{*}$, and determine the
`argmin` indicator $[\text{I}_{l_{r, s} = m_s}]$, where
$m_s = \text{min}(l_{r, s} | r\in\mathcal{R}_{*})$. The distribution
$[\theta_r]$ is obtained as normalized sum of these indicators over
$s=1,\dots, S$. We also need to compute loss values $l_{r_{*}, s}$,
this is done using a cross-validation approximation, see
[here](https://arxiv.org/abs/2201.06834) or our [code](TODO!) for details.
In the beginning, with too little data at the second rung level, we use
$\theta_{r_{min}} = 1$ and 0 elsewhere.

Decisions about a new configuration are based on an acquisition function
over a predictive distribution indexed by $\mathbf{x}$ alone. For
Hyper-Tune, an ensemble distribution with weighting distribution
$[\theta_r]$ is used. Sampling from this distribution works by
first sampling $r\sim [\theta_r]$, then
$f(\mathbf{x}) = f(\mathbf{x}, r)$ from the predictive distribution
for that $r$. This means that models from all rung levels are potentially
used, weighted by how reliable they predict the ordering at the highest
level $r_{*}$ supported by data. In our experiments so far, this
adaptive weighting can outperform the $r_{acq}$ heuristic used in
BOHB and MOBSTER.

Note that our implementation generalizes
[Hyper-Tune](https://arxiv.org/abs/2201.06834) in that ranking losses and
$[\theta_r]$ are estimated once $r_{*} > r_{min}$ (i.e., once
$r_{*}$ is equal to the second rung level). In the original work,
one has to wait until $r_{*} = r_{max}$, i.e. the maximum rung level
is supported by enough data. We find that for many expensive tuning
problems, early decision-making can make a large difference, so if the
Hyper-Tune extensions provide benefits, they should be used as early during
the experiment as possible. For example, in the trial plots for Hyper-Tune
shown above, it takes more than 10000 seconds for 6 trials to reach the
full 200 epochs, so in the original variant of Hyper-Tune, advanced
decision-making only starts when more than half of the experiment is
already done.

If Hyper-Tune is used with more than one bracket, the $[\theta_r]$
is also used in order to sample the bracket for a new trial. To this end,
we need to determine a distribution $P(r)$ over all rung levels
which feature as $r_{min}$ in a bracket. In our NASBench-201
example, if Hyper-Tune is run with 5 brackets, the support of $P(r)$
would be $\mathcal{S} = \{1, 3, 9, 27, 81\}$. Also, denote the
[default distribution](mf_asha.md#asynchronous-hyperband) used in ASHA and
MOBSTER by $P_0(r)$. Let
$r_0 = \text{min}(r_{*}, \text{max}(\mathcal{S}))$. For
$r\in\mathcal{S}$, we define $P(r) = M \theta_r / r$ for
$r\le r_0$, and $P(r) = P_0(r)$ for $r > r_0$, where
$M = \sum_{r\in\mathcal{S}, r\le r_0} P_0(r)$. In other words,
we use $\theta_r / r$ for rung levels supported by data, and the
default $P_0(r)$ elsewhere. Once more, this slightly generalizes
[Hyper-Tune](https://arxiv.org/abs/2201.06834).


In the [next section](mf_comparison.md), we provide some empirical comparison
of all the methods discussed so far.
