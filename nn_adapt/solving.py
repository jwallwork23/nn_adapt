"""
Functions for solving problems defined by configuration
files and performing goal-oriented error estimation.
"""
from firedrake import *
from firedrake.petsc import PETSc
from firedrake.mg.embedded import TransferManager
from pyroteus.error_estimation import get_dwr_indicator


tm = TransferManager()


def get_solutions(mesh, config, solve_adjoint=True, refined_mesh=None):
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
    :return: forward solution, adjoint solution
        and enriched adjoint solution (if requested)
    """

    # Solve forward problem in base space
    V = config.get_function_space(mesh)
    with PETSc.Log.Event("Forward solve"):
        ic = config.get_initial_condition(V)
        solver_obj = config.setup_solver(mesh, ic)
        solver_obj.iterate()
    q = solver_obj.fields.solution_2d
    if not solve_adjoint:
        return q

    # Solve adjoint problem in base space
    with PETSc.Log.Event("Adjoint solve"):
        sp = config.parameters.adjoint_solver_parameters
        J = config.get_qoi(mesh)(q)
        q_star = Function(V)
        F = solver_obj.timestepper.F
        dFdq = derivative(F, q, TrialFunction(V))
        dFdq_transpose = adjoint(dFdq)
        dJdq = derivative(J, q, TestFunction(V))
        solve(dFdq_transpose == dJdq, q_star, solver_parameters=sp)
    if refined_mesh is None:
        return q, q_star

    # Solve adjoint problem in enriched space
    with PETSc.Log.Event("Enrichment"):
        V = config.get_function_space(refined_mesh)
        q_plus = Function(V)
        solver_obj = config.setup_solver(refined_mesh, q_plus)
        q_plus = solver_obj.fields.solution_2d
        J = config.get_qoi(refined_mesh)(q_plus)
        F = solver_obj.timestepper.F
        tm.prolong(q, q_plus)
        q_star_plus = Function(V)
        dFdq = derivative(F, q_plus, TrialFunction(V))
        dFdq_transpose = adjoint(dFdq)
        dJdq = derivative(J, q_plus, TestFunction(V))
        solve(dFdq_transpose == dJdq, q_star_plus, solver_parameters=sp)
    return q, q_star, q_star_plus


def split_into_components(f):
    r"""
    Extend the :attr:`split` method to apply
    to non-mixed :class:`Function`\s.
    """
    return [f] if f.function_space().value_size == 1 else f.split()


def indicate_errors(mesh, config, enrichment_method="h", retall=False):
    """
    Indicate errors according to ``dwr_indicator``,
    using the solver given in the configuration file.

    :arg mesh: input mesh
    :arg config: configuration file, which
        specifies the PDE and QoI
    :kwarg enrichment_method: how to enrich the
        finite element space?
    :kwarg retall: if ``True``, return the forward
        solution and adjoint solution in addition
        to the dual-weighted residual error indicator
    """
    if not enrichment_method == "h":
        raise NotImplementedError  # TODO
    with PETSc.Log.Event("Enrichment"):
        mesh, refined_mesh = MeshHierarchy(mesh, 1)

    # Solve the forward and adjoint problems
    fwd_sol, adj_sol, adj_sol_plus = get_solutions(
        mesh, config, refined_mesh=refined_mesh
    )

    with PETSc.Log.Event("Enrichment"):

        # Prolong
        V_plus = adj_sol_plus.function_space()
        fwd_sol_plg = Function(V_plus)
        tm.prolong(fwd_sol, fwd_sol_plg)
        adj_sol_plg = Function(V_plus)
        tm.prolong(adj_sol, adj_sol_plg)

        # Subtract prolonged adjoint solution from enriched version
        adj_error = Function(V_plus)
        adj_sols_plus = split_into_components(adj_sol_plus)
        adj_sols_plg = split_into_components(adj_sol_plg)
        for i, err in enumerate(split_into_components(adj_error)):
            err += adj_sols_plus[i] - adj_sols_plg[i]

        # Evaluate errors
        dwr = dwr_indicator(config, mesh, fwd_sol_plg, adj_error)

    if retall:
        return dwr, fwd_sol, adj_sol
    else:
        return dwr


def dwr_indicator(config, mesh, q, q_star):
    r"""
    Evaluate the DWR error indicator as a :math:`\mathbb P0` field.

    :arg mesh: the current mesh
    :arg q: the forward solution, transferred into enriched space
    :arg q_star: the adjoint solution in enriched space
    """
    mesh_plus = q.function_space().mesh()

    # Extract indicator in enriched space
    solver_obj = config.setup_solver(mesh_plus, q)
    F = solver_obj.timestepper.F
    V = solver_obj.function_spaces.V_2d
    dwr_plus = get_dwr_indicator(F, q_star, test_space=V)

    # Project down to base space
    P0 = FunctionSpace(mesh, "DG", 0)
    dwr = project(dwr_plus, P0)
    dwr.interpolate(abs(dwr))
    return dwr
