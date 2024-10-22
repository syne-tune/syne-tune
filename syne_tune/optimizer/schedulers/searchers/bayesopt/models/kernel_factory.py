from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel import (
    KernelFunction,
    Matern52,
    ExponentialDecayResourcesKernelFunction,
    ExponentialDecayResourcesMeanFunction,
    FreezeThawKernelFunction,
    FreezeThawMeanFunction,
    CrossValidationMeanFunction,
    CrossValidationKernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.warping import (
    WarpedKernel,
    Warping,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    MeanFunction,
)


SUPPORTED_BASE_MODELS = (
    "matern52-ard",
    "matern52-noard",
)


def base_kernel_factory(name: str, dimension: int, **kwargs) -> KernelFunction:
    assert (
        name in SUPPORTED_BASE_MODELS
    ), f"name = {name} not supported. Choose from:\n{SUPPORTED_BASE_MODELS}"
    return Matern52(
        dimension=dimension,
        ARD=name == "matern52-ard",
        has_covariance_scale=kwargs.get("has_covariance_scale", True),
    )


SUPPORTED_RESOURCE_MODELS = (
    "exp-decay-sum",
    "exp-decay-combined",
    "exp-decay-delta1",
    "freeze-thaw",
    "matern52",
    "matern52-res-warp",
    "cross-validation",
)


def resource_kernel_factory(
    name: str, kernel_x: KernelFunction, mean_x: MeanFunction, **kwargs
) -> (KernelFunction, MeanFunction):
    """
    Given kernel function ``kernel_x`` and mean function ``mean_x`` over config ``x``,
    create kernel and mean functions over ``(x, r)``, where ``r`` is the resource
    attribute (nonnegative scalar, usually in ``[0, 1]``).

    Note: For ``name in ["matern52", "matern52-res-warp"]``, if ``kernel_x`` is
    of type
    :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.warping.WarpedKernel`,
    the resulting kernel inherits this warping.

    :param name: Selects resource kernel type
    :param kernel_x: Kernel function over configs ``x``
    :param mean_x: Mean function over configs ``x``
    :param kwargs: Extra arguments (optional)
    :return: ``(res_kernel, res_mean)``, both over ``(x, r)``
    """
    dim_x = kernel_x.dimension
    if name == "matern52":
        res_kernel = base_kernel_factory("matern52-ard", dimension=dim_x + 1)
        if isinstance(kernel_x, WarpedKernel):
            res_kernel = WarpedKernel(kernel=res_kernel, warpings=kernel_x.warpings)
        res_mean = mean_x
    elif name == "matern52-res-warp":
        # Warping on resource dimension (last one)
        res_warping = Warping(dimension=dim_x + 1, coordinate_range=(dim_x, dim_x + 1))
        if isinstance(kernel_x, WarpedKernel):
            warpings = kernel_x.warpings + [res_warping]
        else:
            warpings = [res_warping]
        res_kernel = WarpedKernel(
            kernel=base_kernel_factory("matern52-ard", dimension=dim_x + 1),
            warpings=warpings,
        )
        res_mean = mean_x
    elif name == "freeze-thaw":
        res_kernel = FreezeThawKernelFunction(kernel_x, mean_x)
        res_mean = FreezeThawMeanFunction(kernel=res_kernel)
    elif name == "cross-validation":
        # ``CrossValidationKernelFunction`` needs two kernels, one over the main
        # effect f(x), the other over the residuals g_k(x). We use ``kernel_x``,
        # ``mean_x`` for the former, and create a ``Matern52`` kernel (no ARD) here
        # for the latter
        num_folds = kwargs.get("num_folds")
        assert (
            num_folds is not None
        ), f"Resource kernel '{name}' needs num_folds argument"
        kernel_residual = base_kernel_factory("matern52-noard", dimension=dim_x)
        res_kernel = CrossValidationKernelFunction(
            kernel_main=kernel_x,
            kernel_residual=kernel_residual,
            mean_main=mean_x,
            num_folds=num_folds,
        )
        res_mean = CrossValidationMeanFunction(kernel=res_kernel)
    else:
        if name == "exp-decay-sum":
            delta_fixed_value = 0.0
        elif name == "exp-decay-combined":
            delta_fixed_value = None
        elif name == "exp-decay-delta1":
            delta_fixed_value = 1.0
        else:
            raise AssertionError("name = '{}' not supported".format(name))
        res_kernel = ExponentialDecayResourcesKernelFunction(
            kernel_x, mean_x, delta_fixed_value=delta_fixed_value
        )
        res_mean = ExponentialDecayResourcesMeanFunction(kernel=res_kernel)

    return res_kernel, res_mean
