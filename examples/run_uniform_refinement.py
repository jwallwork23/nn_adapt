"""
Run a given ``test_case`` of a ``model`` on a sequence of
uniformly refined meshes generated from the initial mesh.
"""
from nn_adapt.solving import *

import argparse
import importlib
import numpy as np
from time import perf_counter


start_time = perf_counter()

# Parse for test case and number of refinements
parser = argparse.ArgumentParser(prog="run_uniform_refinement.py")
parser.add_argument("model", help="The model")
parser.add_argument("test_case", help="The configuration file number")
parser.add_argument("-num_refinements", help="Number of mesh refinements")
parsed_args = parser.parse_args()
model = parsed_args.model
try:
    test_case = int(parsed_args.test_case)
    assert test_case > 0
except ValueError:
    test_case = parsed_args.test_case
num_refinements = int(parsed_args.num_refinements or 5)
assert num_refinements >= 0

# Setup
setup = importlib.import_module(f"{model}.config")
setup.initialise(test_case)
unit = setup.parameters.qoi_unit
mesh = Mesh(f"{model}/meshes/{test_case}.msh")
mh = MeshHierarchy(mesh, num_refinements)

# Run uniform refinement
qois = []
dofs = []
elements = []
setup_time = perf_counter() - start_time
print(f"Test case {test_case}")
print(f"Setup time: {setup_time:.2f} seconds")
for i, mesh in enumerate(mh):
    start_time = perf_counter()
    print(f"  Mesh {i}")
    print(f"    Element count        = {mesh.num_cells()}")
    fwd_sol = get_solutions(mesh, setup, solve_adjoint=False)
    fs = fwd_sol.function_space()
    qoi = assemble(setup.get_qoi(mesh)(fwd_sol))
    print(f"    Quantity of Interest = {qoi} {unit}")
    print(f"    Runtime: {perf_counter() - start_time:.2f} seconds")
    qois.append(qoi)
    dofs.append(sum(fs.dof_count))
    elements.append(mesh.num_cells())
    np.save(f"{model}/data/qois_uniform_{test_case}", qois)
    np.save(f"{model}/data/dofs_uniform_{test_case}", dofs)
    np.save(f"{model}/data/elements_uniform_{test_case}", elements)
print(f"Setup time: {setup_time:.2f} seconds")
