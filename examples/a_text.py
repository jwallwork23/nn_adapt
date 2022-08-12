from copy import deepcopy
from thetis import *
from firedrake.adjoint import *
from firedrake import *
from firedrake_adjoint import *

import numpy as np

lx = 40e3
ly = 2e3
nx = 25
ny = 20
mesh2d = RectangleMesh(nx, ny, lx, ly)


def get_function_space(mesh):
    """
    Construct the (mixed) finite element space used for the
    prognostic solution.
    """
    P1v_2d = get_functionspace(mesh, "DG", 1, vector=True)
    P2_2d = get_functionspace(mesh, "CG", 2)
    return P1v_2d * P2_2d


def get_qoi(mesh):
    """
    Extract the quantity of interest function from the :class:`Parameters`
    object.

    It should have one argument - the prognostic solution.
    """
    def qoi(sol):
        return inner(sol, sol) * ds(2)

    return qoi


P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
depth = 20.0
bathymetry_2d.assign(depth)

# total duration in seconds
t_end = 50
# export interval in seconds
t_export = 10

solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.quadratic_drag_coefficient = Constant(0.0025)

options.swe_timestepper_type = 'CrankNicolson'
options.timestep = 10.0

elev_init = Function(P1_2d, name='initial elevation')

xy = SpatialCoordinate(mesh2d)
gauss_width = 4000.
gauss_ampl = 2.0
gauss_expr = gauss_ampl * exp(-((xy[0]-lx/2)/gauss_width)**2)

elev_init.interpolate(gauss_expr)

tape = get_working_tape()
tape.clear_tape()

# Setup forward solution
solver_obj.assign_initial_conditions(elev=elev_init)
solver_obj.iterate()
fwd_sol = solver_obj.fields.solution_2d

stop_annotating();
solve_blocks = get_solve_blocks()
J_form = inner(fwd_sol, fwd_sol)*ds(2)
J = assemble(J_form)
drag_func = Control(solver_obj.options.quadratic_drag_coefficient)
g = compute_gradient(J, drag_func)

q_star = solve_blocks[0].adj_sol
print(q_star)

# # Adjoint solver
# sp = {
#     "mat_type": "aij",
#     "snes_type": "newtonls",
#     "snes_linesearch_type": "bt",
#     "snes_rtol": 1.0e-08,
#     "snes_max_it": 100,
#     "ksp_type": "preonly",
#     "pc_type": "lu",
#     "pc_factor_mat_solver_type": "mumps",
# }

# V = fwd_sol.function_space()
# q_star = Function(V)
# F = solver_obj.timestepper.F
# sol_temp = Function(V)
# sol_temp.assign(fwd_sol)
# J = get_qoi(mesh2d)(fwd_sol)
# dJdq = derivative(J, fwd_sol, TestFunction(V))
# q_star = []
# for i in range(10):
#     dFdq = derivative(F, sol_temp, TrialFunction(V))
#     dFdq_transpose = adjoint(dFdq)
#     print("this step")
#     solve(dFdq_transpose == dJdq, sol_temp, solver_parameters=sp)
#     q_star.append(sol_temp)


ee_file = File(f"out/adjoint.pvd")
ee_file.write(*q_star.split())

# for i in range(len(q_star)):
#     ee_file.write(*q_star[i].split())
