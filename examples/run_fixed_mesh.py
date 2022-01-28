"""
Run a given ``test_case`` of a ``model`` on the initial mesh alone.
"""
from nn_adapt.solving import *
from firedrake.petsc import PETSc

import argparse
import importlib
from time import perf_counter


start_time = perf_counter()

# Parse for test case
parser = argparse.ArgumentParser(prog='run_fixed_mesh.py')
parser.add_argument('model', help='The model')
parser.add_argument('test_case', help='The configuration file number')
parser.add_argument('-num_refinements', help='Number of mesh refinements')
parser.add_argument('-optimise', help='Turn off plotting and debugging (default False)')
parsed_args, unknown_args = parser.parse_known_args()
model = parsed_args.model
try:
    test_case = int(parsed_args.test_case)
    assert test_case > 0
except ValueError:
    test_case = parsed_args.test_case
num_refinements = int(parsed_args.num_refinements or 0)
assert num_refinements >= 0
optimise = bool(parsed_args.optimise or False)

# Setup
setup = importlib.import_module(f'{model}.config')
setup.initialise(test_case)
unit = setup.parameters.qoi_unit
mesh = Mesh(f'{model}/meshes/{test_case}.msh')
if num_refinements > 0:
    with PETSc.Log.Event('Hierarchy'):
        mesh = MeshHierarchy(mesh, num_refinements)[-1]

# Solve and evaluate QoI
sols = get_solutions(mesh, setup, solve_adjoint=not optimise)
if optimise:
    print(f'QoI for test case {test_case} = {assemble(setup.get_qoi(mesh)(sols)):.2f} {unit}')
else:
    print(f'QoI for test case {test_case} = {assemble(setup.get_qoi(mesh)(sols[0])):.2f} {unit}')
    File(f'{model}/outputs/fixed/forward{test_case}.pvd').write(*sols[0].split())
    File(f'{model}/outputs/fixed/adjoint{test_case}.pvd').write(*sols[1].split())
print(f'  Total time taken: {perf_counter() - start_time:.2f} seconds')
