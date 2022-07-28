"""
Time dependent goal-oriented error estimation
"""
"""
Functions for solving problems defined by configuration
files and performing goal-oriented error estimation.
"""
from firedrake import *
from firedrake.petsc import PETSc
from firedrake.mg.embedded import TransferManager
from firedrake_adjoint import *
from firedrake.adjoint import get_solve_blocks
from pyroteus.error_estimation import get_dwr_indicator
import abc
from time import perf_counter

import matplotlib.pyplot as plt


tm = TransferManager()


class Solver(abc.ABC):
    """
    Base class that defines the API for solver objects.
    """

    @abc.abstractmethod
    def __init__(self, mesh, ic, **kwargs):
        """
        Setup the solver.

        :arg mesh: the mesh to define the solver on
        :arg ic: the initial condition
        """
        pass

    @property
    @abc.abstractmethod
    def function_space(self):
        """
        The function space that the PDE is solved in.
        """
        pass

    @property
    @abc.abstractmethod
    def form(self):
        """
        Return the weak form.
        """
        pass

    @abc.abstractmethod
    def iterate(self, **kwargs):
        """
        Solve the PDE.
        """
        pass

    @property
    @abc.abstractmethod
    def solution(self):
        """
        Return the solution field.
        """
        pass


def get_time_solutions(
    meshes,
    config,
    solve_adjoint=True,
    refined_meshes=None,
    init=None,
    convergence_checker=None,
    **kwargs,
):
    """
    Solve forward and adjoint equations on a
    given mesh.

    This works only for steady-state problems.
    Trying to work it out.

    :arg mesh: input mesh
    :arg config: configuration file, which
        specifies the PDE and QoI
    :kwarg solve_adjoint: should we solve the
        adjoint problem?
    :kwarg refined_mesh: refined mesh to compute
        enriched adjoint solution on
    :kwarg init: custom initial condition function
    :kwarg convergence_checker: :class:`ConvergenceTracer`
        instance
    :return: forward solution, adjoint solution
        and enriched adjoint solution (if requested)
    """
    
    tt_steps = config.parameters.tt_steps

    # Solve forward problem in base space
    V = config.get_function_space(meshes[-1])
    out = {"times": {"forward": -perf_counter()}}
    with PETSc.Log.Event("Forward solve"):
        if init is None:
            ic = config.get_initial_condition(V)
        else:
            ic = init(V)
        solver_obj = config.time_dependent_Solver(meshes, ic=0, **kwargs)
        solver_obj.iterate()
        q = solver_obj.solution
        J = config.get_qoi(V)(q[-1])
        qoi = assemble(J)
        
    out["times"]["forward"] += perf_counter()
    out["qoi"] = qoi
    out["forward"] = q
    if convergence_checker is not None:
        if not convergence_checker.check_qoi(qoi):
            return out
    if not solve_adjoint:
        return out
    
    # Solve adjoint problem in base space
    out["times"]["adjoint"] = -perf_counter()
    with PETSc.Log.Event("Adjoint solve"):
        sp = config.parameters.adjoint_solver_parameters
        adj_solution = []
        dJdu, solve_blocks = solver_obj.adjoint_setup()
        
        for step in range(tt_steps-1):
            adjoint_solution = solve_blocks[step].adj_sol
            adj_solution.append(adjoint_solution)
        
        # initial condition for adjoint solution
        adj_solution.append(dJdu)
        
    out["adjoint"] = adj_solution
    out["times"]["adjoint"] += perf_counter()
    if refined_meshes is None:
        return out

    # Solve adjoint problem in enriched space
    out["times"]["estimator"] = -perf_counter()
    with PETSc.Log.Event("Enrichment"):
        V = config.get_function_space(refined_meshes[-1])
        q_plus = Function(V)
        solver_obj_plus = config.time_dependent_Solver(refined_meshes, q_plus, **kwargs)
        solver_obj_plus.iterate()
        q_plus = solver_obj_plus.solution
        # J = config.get_qoi(refined_mesh[-1])(q_plus[-1])
        adj_solution_plus = []
        dJdu_plus, solve_blocks_plus = solver_obj_plus.adjoint_setup()
        
        for step in range(tt_steps-1):
            adjoint_solution_plus = solve_blocks_plus[step].adj_sol
            adj_solution_plus.append(adjoint_solution_plus)
        
        adj_solution_plus.append(dJdu_plus)
    
    out["enriched_adjoint"] = adj_solution_plus
    out["times"]["estimator"] += perf_counter()
    
    return out


def split_into_components(f):
    r"""
    Extend the :attr:`split` method to apply
    to non-mixed :class:`Function`\s.
    """
    return [f] if f.function_space().value_size == 1 else f.split()


def indicate_time_errors(meshes, config, enrichment_method="h", retall=False, **kwargs):
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
    # with PETSc.Log.Event("Enrichment"):
    mesh_list = []
    ref_mesh_list = []
    tt_steps = len(meshes)
    for i in range(tt_steps):
        mesh, ref_mesh = MeshHierarchy(meshes[i], 1)
        mesh_list.append(mesh)
        ref_mesh_list.append(ref_mesh)

    # Solve the forward and adjoint problems
    out = get_time_solutions(meshes=mesh_list, config=config, refined_meshes=ref_mesh_list, **kwargs)
    if retall and "adjoint" not in out:
        return out

    out["times"]["estimator"] -= perf_counter()
    # with PETSc.Log.Event("Enrichment"):
    adj_sol_plus = out["enriched_adjoint"]
    dwr_list = []
    
    for step in range(tt_steps):
        # Prolong
        V_plus = out["enriched_adjoint"][step].function_space()
        fwd_sol_plg = Function(V_plus)
        tm.prolong(out["forward"][step], fwd_sol_plg)
        adj_sol_plg = Function(V_plus)
        tm.prolong(out["adjoint"][step], adj_sol_plg)

        # Subtract prolonged adjoint solution from enriched version
        adj_error = Function(V_plus)
        adj_sols_plus = split_into_components(out["enriched_adjoint"][step])
        adj_sols_plg = split_into_components(adj_sol_plg)
        for i, err in enumerate(split_into_components(adj_error)):
            err += adj_sols_plus[i] - adj_sols_plg[i]

        # Evaluate errors
        dwr_list.append(dwr_indicator(config, mesh, fwd_sol_plg, adj_error))
    out["dwr"] = dwr_list
        
    out["times"]["estimator"] += perf_counter()

    return out if retall else out["dwr"]


def dwr_indicator(config, mesh, q, q_star):
    r"""
    Evaluate the DWR error indicator as a :math:`\mathbb P0` field.

    :arg mesh: the current mesh
    :arg q: the forward solution, transferred into enriched space
    :arg q_star: the adjoint solution in enriched space
    """
    mesh_plus = q.function_space().mesh()

    # Extract indicator in enriched space
    solver_obj = config.Solver(mesh_plus, q)
    F = solver_obj.form
    V = solver_obj.function_space
    dwr_plus = get_dwr_indicator(F, q_star, test_space=V)

    # Project down to base space
    P0 = FunctionSpace(mesh, "DG", 0)
    dwr = project(dwr_plus, P0)
    dwr.interpolate(abs(dwr))
    return dwr
