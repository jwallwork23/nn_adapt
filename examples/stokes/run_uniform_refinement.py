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
num_refinements = int(parsed_args.num_refinements or 5)
assert num_refinements >= 0

# Setup
setup = importlib.import_module(f'config{test_case}')
field = setup.fields[0]

# Run uniform refinement
qois = []
dofs = []
elements = []
for i in range(num_refinements+1):
    mesh = Mesh(os.path.join(os.path.abspath(os.path.dirname(__file__)), f'meshes/{test_case}.{i}.msh'))
    fwd_sol, mesh_seq = get_solutions(mesh, setup, adjoint=False)
    fs = mesh_seq.function_spaces[field][0]
    qois.append(mesh_seq.J)
    dofs.append(sum(fs.dof_count))
    elements.append(mesh.num_cells())
    np.save(f'data/qois_uniform{test_case}', qois)
    np.save(f'data/dofs_uniform{test_case}', dofs)
    np.save(f'data/elements_uniform{test_case}', elements)
