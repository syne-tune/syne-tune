Implementing Components of Bayesian Optimization
================================================

At this point, you should have obtained an overview of how Bayesian optimization
(BO) is structured in Syne Tune, and understood how a new surrogate model can
be implemented. In this section, we turn to other components of BO: the
*acquisition function*, and the *covariance kernel* of the Gaussian process
surrogate model. We also look inside the factory for creating Gaussian process
based searchers.

Implementing an Acquisition Function
------------------------------------

In Bayesian optimization, the next configuration to sample at is chosen by
minimizing an *acquisition function*:

.. math::
   \mathbf{x}_* = \mathrm{argmin}_{\mathbf{x}} \alpha(\mathbf{x})

In general, the acquisition function :math:`\alpha(\mathbf{x})` is optimized
over encoded vectors :math:`\mathbf{x}`, and the optimal :math:`\mathbf{x}_*`
is rounded back to a configuration. This allows for gradient-based
optimization of :math:`\alpha(\mathbf{x})`.

In Syne Tune, acquisition functions are subclasses of
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.AcquisitionFunction`.
It may depend on one or more surrogate models, by being a function of the
predictive statistics returned by the ``predict`` method of
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.Predictor`.
For a wide range of acquisition functions used in practice, we have that

.. math::
   \alpha(\mathbf{x}) = \alpha(\mu(\mathbf{x}), \sigma(\mathbf{x})).

In other words, :math:`\alpha(\mathbf{x})` is a function of the predictive
mean and standard deviation of a single surrogate model. This case is
covered by
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc.MeanStdAcquisitionFunction`.
More general, this class implements acquisition functions depending on one
or more surrogate models, each of which returning means and (optionally)
standard deviations in ``predict``. Given the generic code in Syne Tune, a
new acquisition function of this type is easy to implement. As an example,
consider the lower confidence bound (LCB) acquisition function:

.. math::
   \alpha_{\mathrm{LCB}}(\mathbf{x}) =
   \mu(\mathbf{x}) - \kappa \sigma(\mathbf{x}),\quad \kappa > 0.

Here is the code:

.. literalinclude:: ../../../../syne_tune/optimizer/schedulers/searchers/bayesopt/models/meanstd_acqfunc_impl.py
   :caption: bayesopt/models/meanstd_acqfunc_impl.py
   :start-at: class LCBAcquisitionFunction(MeanStdAcquisitionFunction):
   :end-before: class EIpuAcquisitionFunction(MeanStdAcquisitionFunction):

* An object is constructed by passing ``model`` (a ``Predictor``) and
  ``kappa`` (the positive constant :math:`\kappa`). The surrogate model
  must return means and standard deviations in its ``predict`` method.
* ``_compute_head``: This method computes
  :math:`\alpha(\mathbf{\mu}, \mathbf{\sigma})`, given means and standard
  deviations. The argument ``output_to_predictions`` is a dictionary of
  dictionaries. If the acquisition function depends on a dictionary of
  surrogate models, the first level corresponds to that. The second level
  corresponds to the statistics returned by ``predict``. In the simple
  case here, the first level is a single entry with key
  :const:`~syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common.INTERNAL_METRIC_NAME`,
  and the second level uses keys "mean" and "std" for means :math:`\mathbf{\mu}`
  and stddevs :math:`\mathbf{\sigma}`. Recall that due to fantasizing, the
  "mean" entry can be a ``(n, nf)`` matrix, in which case we compute the
  average along the columns. The argument ``current_best`` is needed only
  for acquisition functions which depend on the incumbent.
* ``_compute_head_and_gradient``: This method is needed for the computation
  of :math:`\partial\alpha/\partial\mathbf{x}`, for a single input
  :math:`\mathbf{x}`. Given the same arguments
  as ``_compute_head`` (but for :math:`n = 1` inputs), it returns a
  ``HeadWithGradient`` object, whose ``hval`` entry is the same as the
  return value of ``_compute_head``, whereas the ``gradient`` entry contains
  the head gradients which are passed to the ``backward_gradient`` method of
  the
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.Predictor`.
  This entry is a nested dictionary of the same structure as
  ``output_to_predictions``. The head gradient for a single surrogate model
  (as in our example) has :math:`\partial\alpha/(\partial\mathbf{\mu})` for
  "mean" and :math:`\partial\alpha/(\partial\mathbf{\sigma})` for "std".
  It is particularly simple for the LCB example.
* ``_head_needs_current_best`` returns ``False``, since the LCB acquisition
  function does not depend on the incumbent (i.e., the current best metric
  value), which means that the ``current_best`` arguments need not be
  provided.

A slightly more involved example is
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc_impl.EIAcquisitionFunction`,
representing the expected improvement (EI) acquisition function, which is the
default choice for :class:`~syne_tune.optimizer.baselines.BayesianOptimization`
in Syne Tune. This function depends on the incumbent, so ``current_best`` needs
to be given. Note that if the means passed to ``_compute_head`` have shape
``(n, nf)`` due to fantasies, then ``current_best`` has shape ``(1, nf)``,
since the incumbent depends on the fantasy sample.

Acquisition functions can depend on more than one surrogate model. In such a
case, the ``model`` argument to their constructor is a dictionary, and the
key names of the corresponding models (or outputs) are also used in the
``output_to_predictions`` arguments and head gradients:

* :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc_impl.EIpuAcquisitionFunction`
  is an acquisition function for cost-aware HPO:

  .. math::
     \alpha_{\mathrm{EIpu}}(\mathbf{x}) =
     \frac{\alpha_{\mathrm{EI}}(\mu_y(\mathbf{x}), \sigma_y(\mathbf{x}))}{\mu_c(\mathbf{x})^{\rho}}

  Here, :math:`(\mu_y, \sigma_y)` are predictions from the surrogate model for
  the target function :math:`y(\mathbf{x})`, whereas :math:`\mu_c` are mean
  predictions for the cost function :math:`c(\mathbf{x})`. The latter can be
  represented by a deterministic surrogate model, whose ``predict`` method only
  returns means as "mean". In fact, the method ``_output_to_keys_predict``
  specifies which moments are required from each surrogate model.
* :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc_impl.CEIAcquisitionFunction`
  is an acquisition function for constrained HPO:

  .. math::
     \alpha_{\mathrm{CEI}}(\mathbf{x}) =
     \alpha_{\mathrm{EI}}(\mu_y(\mathbf{x}), \sigma_y(\mathbf{x})) \cdot
     \mathbb{P}(c(\mathbf{x})\le 0).

  Here, :math:`y(\mathbf{x})` is the target function, :math:`c(\mathbf{x})` is
  the constraint function. Both functions are represented by probabilistic
  surrogate models, whose ``predict`` method returns means and stddevs.
  We say that :math:`\mathbf{x}` is feasible if :math:`c(\mathbf{x})\le 0`,
  and the goal is to minimize :math:`y(\mathbf{x})` over feasible points.

  One difficulty with this acquisition function is that the incumbent in
  the EI term is computed only over observations which are feasible (so
  :math:`c_i\le 0`). This means we cannot rely on the surrogate model for
  :math:`y(\mathbf{x})` to provide the incumbent, but instead need to determine
  the feasible incumbent ourselves, in the ``_get_current_bests_internal``
  method.

A final complication in
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc.MeanStdAcquisitionFunction`
arises if some or all surrogate models are MCMC ensembles. In such a case,
we average over the sample for each surrogate model involved. Inside this sum
over the Cartesian product, the incumbent depends on the sample index for each
model. This is dealt with by
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc.CurrentBestProvider`.
In the default case for an acquisition function which needs the incumbent
(such as, for example, EI), this value depends only on the model for the
active (target) metric, and
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc.ActiveMetricCurrentBestProvider`
is used.

.. note::
   Acquisition function implementations are independent of which
   auto-differentiation mechanism is used under the hood. Different to
   surrogate models, there is no acquisition function code in
   :mod:`syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd`.
   This is because the implementation only needs to provide head gradients
   in ``compute_acq_with_gradient``, which are easy to derive and compute
   for common acquisition functions.

Implementing a Covariance Function for GP Surrogate Models
----------------------------------------------------------

A Gaussian process, modelling a random function :math:`y(\mathbf{x})`, is
defined by a mean function :math:`\mu(\mathbf{x})` and a covariance function
(or kernel) :math:`k(\mathbf{x}, \mathbf{x}')`. While Syne Tune contains a
number of different covariance functions for multi-fidelity HPO, where
learning curves :math:`y(\mathbf{x}, r)` are modelled, :math:`r = 1, 2, \dots`
the number of epochs trained (details are provided
`here <../multifidelity/mf_async_model.html#surrogate-models-of-learning-curves>`__),
it currently provides the Matern 5/2 covariance function only for models
of :math:`y(\mathbf{x})`. A few comments up front:

* Mean and covariance functions are parts of (Gaussian process) surrogate
  models. For these models, complex gradients are required for different
  purposes. First, our Bayesian optimization code supports gradient-based
  minimization of the acquisition function. Second, a surrogate model is
  fitted to observed data, which is typically done by gradient-based
  optimization (e.g., marginal likelihood optimization, empirical Bayes)
  or by gradient-based Markov Chain Monte Carlo (e.g., Hamiltonian Monte Carlo).
  This means that covariance function code must be written in a framework
  supporting automatic differentiation. In Syne Tune, this code resides in
  :mod:`syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd`. It is
  based on `autograd <https://github.com/HIPS/autograd>`__.
* Covariance functions contain parameters to be fitted to observed data.
  Kernels in Syne Tune typically feature an overall output scale, as well
  as inverse bandwidths for the input. In the (so called) *automatic relevance
  determination* parameterization, we use one inverse bandwidth per input
  vector component. This allows the surrogate model to learn relevance to
  certain input components: if components are not relevant to explain the
  observed data, their inverse bandwidths can be driven to very small values.
  Syne Tune uses code extracted from
  `MXNet Gluon <https://mxnet.apache.org/versions/1.6/api/python/docs/api/gluon/index.html>`__
  for managing parameters. The base class
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel.base.KernelFunction`
  derives from
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean.MeanFunction`,
  which derives from
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon.Block`.
  The main service of this class is to maintain a parameter dictionary,
  collecting all parameters in the current objects and its members (recursively).

In order to understand how a new covariance function can be implemented, we will
go through the most important parts of
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel.base.Matern52`.
This covariance function is defined as:

.. math::
   k(\mathbf{x}, \mathbf{x}') = c \left( 1 + d + d^2/3 \right) e^{-d}, \quad
   d = \sqrt{5} \|\mathbf{S} (\mathbf{x} - \mathbf{x}')\|.

Its parameters are the output scale :math:`c > 0` and the inverse bandwidths
:math:`s_j > 0`, where :math:`\mathbf{S}` is the
diagonal matrix with diagonal entries :math:`s_j`. If ``ARD == False``, there
is only a single bandwidth parameter :math:`s > 0`.

First, we need some includes:

.. literalinclude:: ../../../../syne_tune/optimizer/schedulers/searchers/bayesopt/gpautograd/kernel/base.py
   :caption: bayesopt/gpautograd/kernel/base.py -- includes
   :start-after: # permissions and limitations under the License.
   :end-before: class KernelFunction(MeanFunction):

Since a number of covariance functions are simple expressions of squared
distances :math:`\|\mathbf{S} (\mathbf{x} - \mathbf{x}')\|^2`, Syne Tune contains
a block for this one:

.. literalinclude:: ../../../../syne_tune/optimizer/schedulers/searchers/bayesopt/gpautograd/kernel/base.py
   :caption: bayesopt/gpautograd/kernel/base.py -- SquaredDistance
   :start-at: class SquaredDistance(Block):
   :end-before: class Matern52(KernelFunction):

* In the constructor, we create a parameter vector for the inverse bandwidths
  :math:`[s_j]`, which can be just a scalar if ``ARD == False``. In Syne Tune,
  each parameter has an encoding (e.g., identity or logarithmic), which
  includes a lower and upper bound, an initial value, as well as a prior
  distribution. The latter is used for regularization during optimization.
* The most important method is ``forward``. Given two matrices
  :math:`\mathbf{X}_1`, :math:`\mathbf{X}_2`, whose rows are input vectors,
  we compute the matrix :math:`[\|\mathbf{x}_{1:i} - \mathbf{x}_{2:j}\|^2]_{i, j}`
  of squared distances. Most important, we use ``anp = autograd.numpy`` here
  instead of ``numpy``. These ``autograd`` wrappers ensure that automatic
  differentiation can be used in order to compute gradients w.r.t. leaf nodes
  in the computation graph spanned by the ``numpy`` computations. Also, note
  the use of ``encode_unwrap_parameter`` in ``_inverse_bandwidths`` to obtain
  the inverse bandwidth parameters as ``numpy`` array. Finally, note that
  ``X1`` and ``X2`` can be the same object, in which case we can save compute
  time and create a smaller computation graph.
* Each block in Syne Tune also provides ``get_params`` and ``set_params``
  methods, which are used for serialization and deserialization.

Given this code, the implementation of
:class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel.base.Matern52`
is simple:

.. literalinclude:: ../../../../syne_tune/optimizer/schedulers/searchers/bayesopt/gpautograd/kernel/base.py
   :caption: bayesopt/gpautograd/kernel/base.py -- Matern52
   :start-at: class Matern52(KernelFunction):

* In the constructor, we create an object of type
  :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel.base.SquaredDistance`.
  A nice feature of MXNet Gluon blocks is that the parameter dictionary of an
  object is automatically extended by the dictionaries of members, so we don't
  need to cater for that. Beware that this only works for members which are of
  type ``Block`` directly. If you use a list or dictionary containing such
  objects, you need to include their parameter dictionaries explicitly.
  Next, we also define a covariance scale parameter :math:`c > 0`, unless
  ``has_covariance_scale == False``.
* ``forward`` calls ``forward`` of the ``SquaredDistance`` object, then
  computes the kernel matrix, using ``anp = autograd.numpy`` once more.
* ``diagonal`` returns the diagonal of the kernel matrix based on a
  matrix ``X`` of inputs. For this particular kernel, the diagonal does not
  depend on the content of ``X``, but only its shape, which is why
  ``diagonal_depends_on_X`` returns ``False``.
* Besides ``get_params`` and ``set_params``, we also need to implement
  ``param_encoding_pairs``, which is required by the optimization code
  used for fitting the surrogate model parameters.

At this point, you should not have any major difficulties implementing a new
covariance function, such as the Gaussian kernel or the Matern kernel with
parameter 3/2.

The Factory for Gaussian Process Searchers
------------------------------------------

Once a covariance function (or any other component of a surrogate model) has
been added, how is it accessed by a user? In general, all details about the
surrogate model are specified in ``search_options`` passed to
:class:`~syne_tune.optimizer.schedulers.FIFOScheduler` or
:class:`~syne_tune.optimizer.baselines.BayesianOptimization`. Available options
are documented in
:class:`~syne_tune.optimizer.schedulers.searchers.GPFIFOSearcher`. Syne Tune
offers a range of searchers based on various Gaussian process surrogate models
(e.g., single fidelity, multi-fidelity, constrained, cost-aware). The code to
generate all required components for these searchers is bundled in
:mod:`~syne_tune.optimizer.schedulers.searchers.gp_searcher_factory`. For
each type of searcher, there is a factory function and a defaults function.
For :class:`~syne_tune.optimizer.baselines.BayesianOptimization` (which is
equivalent to :class:`~syne_tune.optimizer.schedulers.FIFOScheduler` with
``searcher="bayesopt"``), we have:

* :func:`~syne_tune.optimizer.schedulers.searchers.gp_searcher_factory.gp_fifo_searcher_factory`:
  Takes ``search_options`` for ``kwargs`` and returns the arguments for the
  :class:`~syne_tune.optimizer.schedulers.searchers.GPFIFOSearcher` constructor.
* :func:`~syne_tune.optimizer.schedulers.searchers.gp_searcher_factory.gp_fifo_searcher_defaults`:
  Provides default values and type constraints for ``search_options``

The searcher object is created in
:func:`~syne_tune.optimizer.schedulers.searchers.searcher_factory.searcher_factory`.
Finally, ``search_options`` are merged with default values, and ``searcher_factory``
is called in the constructor of
:class:`~syne_tune.optimizer.schedulers.FIFOScheduler`. This process keeps
things simple for the user, who just has to specify the type of searcher by
``searcher``, and additional arguments by ``search_options``. For any argument
not provided there, a sensible default value is used.

Factory and default functions in
:mod:`~syne_tune.optimizer.schedulers.searchers.gp_searcher_factory` are based
on common code in this module, which reflects the complexity of some of the
searchers, but is otherwise self-explanatory. As a continuation of the
previous section, suppose we had implemented a novel covariance function to
be used in GP-based Bayesian optimization. The user-facing argument to select
a kernel is ``gp_base_kernel``, its default value is "matern52-ard" (Matern
5/2 with ARD parameters). Here is the code for creating this covariance
function in :mod:`~syne_tune.optimizer.schedulers.searchers.gp_searcher_factory`:

.. literalinclude:: ../../../../syne_tune/optimizer/schedulers/searchers/gp_searcher_factory.py
   :caption: gp_searcher_factory.py
   :start-at: def _create_base_gp_kernel(hp_ranges: HyperparameterRanges, **kwargs) -> KernelFunction:
   :end-before: def _create_gp_common(hp_ranges: HyperparameterRanges, **kwargs):

* Ignoring ``transfer_learning_task_attr``, we first call ``base_kernel_factory``
  to create the base kernel, passing ``kwargs["gp_base_kernel"]`` as its name.
* Syne Tune also supports warping of the inputs to a kernel, which adds two
  more parameters for each component (except those coming from categorical
  hyperparameters, these are not warped).

.. literalinclude:: ../../../../syne_tune/optimizer/schedulers/searchers/bayesopt/models/kernel_factory.py
   :caption: bayesopt/models/kernel_factory.py
   :start-after: # permissions and limitations under the License.
   :end-before: SUPPORTED_RESOURCE_MODELS = (

* ``base_kernel_factory`` creates the base kernel, based on its name (must be
  in ``SUPPORTED_BASE_MODELS``, the dimension of input vectors, as well as
  further parameters (``has_covariance_scale`` in our example). Currently,
  Syne Tune only supports the Matern  5/2 kernel, with and without ARD.
* Had we implemented a novel covariance function, we would have to select a
  new name, insert it into ``SUPPORTED_BASE_MODELS``, and insert code into
  ``base_kernel_factory``. Once this is done, the new base kernel can as well
  be selected as component in multi-fidelity or constrained Bayesian
  optimization.
