from nn_adapt.features import *
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
from time import perf_counter

import matplotlib.pyplot as plt

tt_steps = 10

setup1 = importlib.import_module(f"burgers_n2n.config")
meshes = [UnitSquareMesh(20, 20) for _ in range(tt_steps)]
out1 = indicate_errors_n2n(meshes=meshes, config=setup1)
print(out1)

mesh = UnitSquareMesh(20, 20)
setup2 = importlib.import_module(f"burgers_one2n.config")
out2 = indicate_errors_one2n(mesh=mesh, config=setup2)
print(out2)

# fig, axes = plt.subplots(10,2)

# for i in range(tt_steps):
#     tricontourf(out['forward'][i], axes=axes[i][0])
#     tricontourf(out['adjoint'][i], axes=axes[i][1])

# plt.savefig("test1.jpg")


# mesh = UnitSquareMesh(30, 30)
# setup2 = importlib.import_module(f"burgers.config")
# out = get_solutions(mesh=mesh, config=setup2)
# fig, axes = plt.subplots(2)
# tricontourf(out['forward'], axes=axes[0])
# tricontourf(out['adjoint'], axes=axes[1])
    
# plt.savefig("test2.jpg")

