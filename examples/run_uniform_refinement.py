from nn_adapt.solving import *

import argparse
import importlib
import numpy as np
import os
from time import perf_counter


start_time = perf_counter()

# Parse for test case and number of refinements
parser = argparse.ArgumentParser()
parser.add_argument('model', help='The model')
parser.add_argument('test_case', help='The configuration file number')
parser.add_argument('-num_refinements', help='Number of mesh refinements')
parsed_args = parser.parse_args()
model = parsed_args.model
assert model in ['turbine']
test_case = int(parsed_args.test_case)
assert test_case in list(range(16))
num_refinements = int(parsed_args.num_refinements or 5)
assert num_refinements >= 0

# Setup
setup = importlib.import_module(f'{model}.config{test_case}')
field = setup.fields[0]
mesh = Mesh(f'{os.path.abspath(os.path.dirname(__file__))}/{model}/meshes/{test_case}.msh')
mh = MeshHierarchy(mesh, num_refinements)

# Run uniform refinement
qois = []
dofs = []
elements = []
setup_time = perf_counter() - start_time
print(f'Test case {test_case}')
print(f'Setup time: {setup_time:.2f} seconds')
for i, mesh in enumerate(mh):
    start_time = perf_counter()
    print(f'  Mesh {i}')
    print(f'    Element count        = {mesh.num_cells()}')
    fwd_sol = get_solutions(mesh, setup, solve_adjoint=False)
    fs = fwd_sol.function_space()
    qoi = assemble(setup.get_qoi(mesh)(fwd_sol))
    print(f'    Quantity of Interest = {qoi}')
    print(f'    Runtime: {perf_counter() - start_time:.2f} seconds')
    qois.append(qoi)
    dofs.append(sum(fs.dof_count))
    elements.append(mesh.num_cells())
    np.save(f'{model}/data/qois_uniform_{test_case}', qois)
    np.save(f'{model}/data/dofs_uniform_{test_case}', dofs)
    np.save(f'{model}/data/elements_uniform_{test_case}', elements)
print(f'Setup time: {setup_time:.2f} seconds')
