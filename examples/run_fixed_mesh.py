from nn_adapt.solving import *
from firedrake.petsc import PETSc

import argparse
import importlib
import os
from time import perf_counter


start_time = perf_counter()

# Parse for test case
parser = argparse.ArgumentParser()
parser.add_argument('model', help='The model')
parser.add_argument('test_case', help='The configuration file number')
parser.add_argument('-num_refinements', help='Number of mesh refinements')
parser.add_argument('-optimise', help='Turn off plotting and debugging (default False)')
parsed_args, unknown_args = parser.parse_known_args()
model = parsed_args.model
assert model in ['stokes', 'turbine']
test_case = int(parsed_args.test_case)
assert test_case in list(range(12))
num_refinements = int(parsed_args.num_refinements or 0)
assert num_refinements >= 0
optimise = bool(parsed_args.optimise or False)

# Setup
setup = importlib.import_module(f'{model}.config{test_case}')
mesh = Mesh(f'{os.path.abspath(os.path.dirname(__file__))}/{model}/meshes/{test_case}.msh')
if num_refinements > 0:
    with PETSc.Log.Event('Hierarchy'):
        mesh = MeshHierarchy(mesh, num_refinements)[-1]

# Solve and evaluate QoI
fwd_sol = get_solutions(mesh, setup, solve_adjoint=False)
J = assemble(setup.get_qoi(mesh)(fwd_sol))
print(f'QoI for test case {test_case} = {J:.2f} MW')
if not optimise:
    File(f'{model}/outputs/fixed/forward{test_case}.pvd').write(*fwd_sol.split())
    File(f'{model}/outputs/fixed/adjoint{test_case}.pvd').write(*adj_sol.split())
print(f'  Total time taken: {perf_counter() - start_time:.2f} seconds')
