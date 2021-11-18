from nn_adapt import *
import argparse
import importlib
import numpy as np


# Parse for test case and number of refinements
parser = argparse.ArgumentParser()
parser.add_argument('test_case', help='The configuration file number')
parser.add_argument('-num_refinements', help='Number of mesh refinements')
parsed_args = parser.parse_args()
test_case = int(parsed_args.test_case)
assert test_case in [0, 1, 2, 3, 4]
num_refinements = int(parsed_args.num_refinements or 6)
assert num_refinements >= 0

# Setup mesh hierarchy
setup = importlib.import_module(f'config{test_case}')
field = setup.fields[0]
mh = MeshHierarchy(setup.mesh, num_refinements)

# Run uniform refinement
qois = []
dofs = []
elements = []
for mesh in mh:
    fwd_sol, mesh_seq = get_solutions(mesh, setup, adjoint=False)
    fs = mesh_seq.function_spaces[field][0]
    qois.append(mesh_seq.J)
    dofs.append(sum(fs.dof_count))
    elements.append(mesh.num_cells())
    np.save(f'data/qois_uniform{test_case}', qois)
    np.save(f'data/dofs_uniform{test_case}', dofs)
    np.save(f'data/elements_uniform{test_case}', elements)
