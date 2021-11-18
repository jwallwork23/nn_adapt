from firedrake import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('test_case')
parser.add_argument('-cell_size')
parsed_args, unknown_args = parser.parse_known_args()
test_case = int(parsed_args.test_case)
assert test_case in [0, 1, 2, 3, 4]
cell_size = float(parsed_args.cell_size or 1.0)
assert cell_size > 0.0

mesh = Mesh(f'{test_case}.3.msh')
P1_ten = TensorFunctionSpace(mesh, 'CG', 1)
metric = interpolate(cell_size*Identity(2), P1_ten)
newmesh = adapt(mesh, metric)
File(f'mesh{test_case}.pvd').write(newmesh.coordinates)
viewer = PETSc.Viewer().createHDF5(f'{test_case}.h5', 'w')
viewer(newmesh.topology_dm)
