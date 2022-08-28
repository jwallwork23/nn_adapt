"""
Functions for solving problems defined by configuration
files and performing goal-oriented error estimation.
"""
from firedrake import *
from firedrake_adjoint import *


def get_solutions(
    mesh,
    config,
    solve_adjoint=True,
    convergence_checker=None,
    **kwargs,
):
    """
    Solve forward and adjoint equations on a
    given mesh.

    This works only for steady-state problems.

    :arg mesh: input mesh
    :arg config: configuration file, which
        specifies the PDE and QoI
    :kwarg solve_adjoint: should we solve the
        adjoint problem?
    :kwarg refined_mesh: refined mesh to compute
        enriched adjoint solution on
    :kwarg convergence_checker: :class:`ConvergenceTracer`
        instance
    :return: forward solution, adjoint solution
        and enriched adjoint solution (if requested)
    """
    out = {}
    solve_adjoint = True  # TODO: Account for the false case!
    # NOTE: None of the timings will work!

    # Solve forward problem in base space
    mesh_seq = config.GoalOrientedMeshSeq(mesh)
    solutions = mesh_seq.solve_adjoint()
    fields = mesh_seq.fields
    qoi = mesh_seq.J
    out["qoi"] = qoi
    out["forward"] = {f: solutions[f]["forward"] for f in fields}
    if convergence_checker is not None:
        if convergence_checker.check_qoi(qoi):
            return out
    else:
        print("No convergence checker")
    if not solve_adjoint:
        return out

    # Solve adjoint problem in base space
    out["adjoint"] = {f: solutions[f]["adjoint"] for f in fields}
    return out


def split_into_components(f):
    r"""
    Extend the :attr:`split` method to apply
    to non-mixed :class:`Function`\s.
    """
    return [f] if f.function_space().value_size == 1 else f.split()


def indicate_errors(mesh, config, enrichment_method="h", retall=False, **kwargs):
    """
    Indicate errors according to the ``GoalOrientedMeshSeq``
    given in the configuration file.

    :arg mesh: input mesh
    :arg config: configuration file, which
        specifies the PDE and QoI
    :kwarg enrichment_method: how to enrich the
        finite element space?
    :kwarg retall: if ``True``, return the forward
        solution and adjoint solution in addition
        to the dual-weighted residual error indicator
    """
    out = {}
    if not enrichment_method == "h":
        raise NotImplementedError  # TODO
    mesh_seq = config.GoalOrientedMeshSeq(mesh)
    fields = mesh_seq.fields
    kw = {"enrichment_method": enrichment_method}
    solutions, indicators = mesh_seq.indicate_errors(enrichment_kwargs=kw)
    integrated = []
    for i, indicator in enumerate(indicators):
        dt = mesh_seq.time_partition.timesteps[i]
        contrib = Function(indicator[0].function_space())
        for indi in indicator:
            contrib += dt * indi
        integrated.append(contrib)
    qoi = mesh_seq.J
    out["qoi"] = qoi
    out["forward"] = {f: solutions[f]["forward"] for f in fields}
    out["adjoint"] = {f: solutions[f]["adjoint"] for f in fields}
    out["dwr"] = integrated
    return out if retall else out["dwr"]
