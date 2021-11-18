from nn_adapt import *
import argparse
import importlib
import os


# Parse for test case
parser = argparse.ArgumentParser()
parser.add_argument('model', help='The model')
parser.add_argument('test_case', help='The configuration file number')
parsed_args = parser.parse_args()
model = parsed_args.model
assert model in ['stokes']
test_case = int(parsed_args.test_case)
assert test_case in [0, 1, 2, 3, 4]

# Run fixed mesh setup
setup = importlib.import_module(f'{model}.config{test_case}')
mesh = Mesh(f'{os.path.abspath(os.path.dirname(__file__))}/{model}/meshes/{test_case}.0.msh')
fwd_sol, adj_sol, mesh_seq = get_solutions(mesh, setup)
print(f'QoI for test case {test_case} = {mesh_seq.J}')
File(f'{model}/outputs/fixed/forward{test_case}.pvd').write(*fwd_sol.split())
File(f'{model}/outputs/fixed/adjoint{test_case}.pvd').write(*adj_sol.split())
