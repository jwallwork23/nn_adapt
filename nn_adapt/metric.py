"""
Functions for generating Riemannian metrics from solution
fields.
"""
from pyroteus import *
from nn_adapt.features import split_into_scalars
from nn_adapt.solving import *
from firedrake.meshadapt import RiemannianMetric


def get_hessians(f, **kwargs):
    """
    Compute Hessians for each component of
    a :class:`Function`.

    Any keyword arguments are passed to
    ``recover_hessian``.

    :arg f: the function
    :return: list of Hessians of each
        component
    """
    kwargs.setdefault("method", "Clement")
    return tuple(
        space_normalise(hessian_metric(recover_hessian(fij, **kwargs)), 4000.0, "inf")
        for i, fi in split_into_scalars(f).items()
        for fij in fi
    )


def go_metric(
    mesh,
    config,
    enrichment_method="h",
    target_complexity=4000.0,
    average=True,
    interpolant="Clement",
    anisotropic=False,
    retall=False,
    convergence_checker=None,
    **kwargs,
):
    """
    Compute an anisotropic goal-oriented
    metric field, based on a mesh and
    a configuration file.

    :arg mesh: input mesh
    :arg config: configuration file, which
        specifies the PDE and QoI
    :kwarg enrichment_method: how to enrich the
        finite element space?
    :kwarg target_complexity: target complexity
        of the goal-oriented metric
    :kwarg average: should the Hessian components
        be combined using averaging (or intersection)?
    :kwarg interpolant: which method to use to
        interpolate into the target space?
    :kwarg anisotropic: toggle isotropic vs.
        anisotropic metric
    :kwarg h_min: minimum magnitude
    :kwarg h_max: maximum magnitude
    :kwarg a_max: maximum anisotropy
    :kwarg retall: if ``True``, the error indicator,
        forward solution and adjoint solution
        are returned, in addition to the metric
    :kwarg convergence_checker: :class:`ConvergenceTracer`
        instance
    """
    h_min = kwargs.pop("h_min", 1.0e-30)
    h_max = kwargs.pop("h_max", 1.0e+30)
    a_max = kwargs.pop("a_max", 1.0e+30)
    out = indicate_errors(
        mesh,
        config,
        enrichment_method=enrichment_method,
        retall=True,
        convergence_checker=convergence_checker,
        **kwargs,
    )
    if retall and "adjoint" not in out:
        return out
    out["estimator"] = out["dwr"][0].vector().gather().sum()  # FIXME: Only uses 0th
    if convergence_checker is not None:
        if convergence_checker.check_estimator(out["estimator"]):
            return out

    with PETSc.Log.Event("Metric construction"):
        if anisotropic:
            field = list(out["forward"].keys())[0]
            fwd = out["forward"][field][0]  # FIXME: Only uses 0th
            hessians = sum([get_hessians(sol) for sol in fwd], start=())
            hessian = combine_metrics(*hessians, average=average)
        else:
            hessian = None
        metric = anisotropic_metric(
            out["dwr"][0],  # FIXME: Only uses 0th
            hessian=hessian,
            target_complexity=target_complexity,
            target_space=TensorFunctionSpace(mesh, "CG", 1),
            interpolant=interpolant,
        )
        space_normalise(metric, target_complexity, "inf")
        enforce_element_constraints(metric, h_min, h_max, a_max)
        out["metric"] = RiemannianMetric(mesh)
        out["metric"].assign(metric)
    return out if retall else out["metric"]
