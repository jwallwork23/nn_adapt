from nn_adapt.features import *
from nn_adapt.features import extract_array
from nn_adapt.metric import *
from nn_adapt.parse import Parser
from nn_adapt.solving_one2n import *
from nn_adapt.solving_n2n import *
from nn_adapt.solving import *
from nn_adapt.utility import ConvergenceTracker
from firedrake.meshadapt import adapt
from firedrake.petsc import PETSc

import importlib
import numpy as np

tt_steps = 10

# setup1 = importlib.import_module(f"burgers_n2n.config")
# meshes = [UnitSquareMesh(20, 20) for _ in range(tt_steps)]
# out1 = indicate_errors_n2n(meshes=meshes, config=setup1)
# print(out1)

# mesh = UnitSquareMesh(20, 20)
# setup2 = importlib.import_module(f"burgers_one2n.config")
# out2 = indicate_errors_one2n(mesh=mesh, config=setup2)
# print(out2)

mesh = UnitSquareMesh(20, 20)
setup2 = importlib.import_module(f"burgers_one2n.config")
out2 = get_solutions_one2n(mesh=mesh, config=setup2)
fwd_sol = out2["forward"]


# Adjoint solver
sp = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_rtol": 1.0e-08,
    "snes_max_it": 100,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

V = fwd_sol[-1].function_space()
q_star = Function(V)
F = setup2.Solver_one2n(mesh=mesh, ic=0, config=setup2).form
sol_temp = Function(V)
sol_temp.assign(fwd_sol[-1])
J = setup2.get_qoi(mesh)(fwd_sol[-1])
dJdq = derivative(J, fwd_sol[-1], TestFunction(V))
q_star = []
for i in range(1, 11):
    V = fwd_sol[-i].function_space()
    q_star = Function(V)
    dFdq = derivative(F, fwd_sol[-i], TrialFunction(V))
    print(dFdq)
    dFdq_transpose = adjoint(dFdq)
    print("this step")
    solve(dFdq_transpose == dJdq, q_star, solver_parameters=sp)
    q_star.append(sol_temp)


ee_file = File(f"out/adjoint.pvd")
# ee_file.write(*q_star.split())

for i in range(len(q_star)):
    ee_file.write(*q_star[i].split())

