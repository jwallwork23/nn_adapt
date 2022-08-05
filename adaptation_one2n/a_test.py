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

# mesh = UnitSquareMesh(20, 20)
# setup2 = importlib.import_module(f"burgers_one2n.config")
# out2 = get_solutions_one2n(mesh=mesh, config=setup2)
# test_array = time_integrate(out2["forward"])
# print(extract_array(test_array, centroid=True))

a = None
b = 1
print(a or b)

