from nn_adapt import *
import argparse
import importlib
import os


# Parse for test case
parser = argparse.ArgumentParser()
parser.add_argument('test_case', help='The configuration file number')
test_case = int(parser.parse_args().test_case)
assert test_case in [0, 1, 2, 3, 4]

# Run fixed mesh setup
setup = importlib.import_module(f'config{test_case}')
fwd_sol, adj_sol, mesh_seq = get_solutions(setup.mesh, setup)
print(f'QoI for test case {test_case} = {mesh_seq.J}')
File(f'outputs/fixed/forward{test_case}.pvd').write(*fwd_sol.split())
File(f'outputs/fixed/adjoint{test_case}.pvd').write(*adj_sol.split())
