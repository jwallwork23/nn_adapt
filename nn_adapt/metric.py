from pyroteus import *
from nn_adapt.solving import *


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
    kwargs.setdefault('method', 'Clement')
    return [
        space_normalise(hessian_metric(recover_hessian(fij, **kwargs)), 4000.0, 'inf')
        for i, fi in split_into_scalars(f).items()
        for fij in fi
    ]


def go_metric(mesh, config, enrichment_method='h', target_complexity=4000.0,
              average=False, interpolant='L2', anisotropic=False, retall=False):
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
    :kwarg retall: if ``True``, the error indicator,
        forward solution and adjoint solution
        are returned, in addition to the metric
    """
    dwr, fwd_sol, adj_sol = indicate_errors(
        mesh, config, enrichment_method=enrichment_method, retall=True
    )
    with PETSc.Log.Event('Metric construction'):
        if anisotropic:
            hessian = combine_metrics(*get_hessians(fwd_sol), average=average)
        else:
            hessian = None
        metric = anisotropic_metric(
            dwr, hessian=hessian,
            target_complexity=target_complexity,
            target_space=TensorFunctionSpace(mesh, 'DG', 0),
            interpolant=interpolant
        )
    if retall:
        return metric, dwr, fwd_sol, adj_sol
    else:
        return metric
