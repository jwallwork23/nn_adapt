from firedrake import *
from plotting import *

import argparse
import importlib
import os


matplotlib.rcParams['font.size'] = 12

# Parse model
parser = argparse.ArgumentParser(prog='plot_progress.py')
parser.add_argument('model', help='The equation set being solved')
args = parser.parse_args()
model = args.model
assert model in ['stokes', 'turbine']

# Bounding box
xmin = 0
xmax = 1200
ymin = 0
ymax = 500
eps = 5

fig, axes = plt.subplots(ncols=3, nrows=4, figsize=(10, 7))
for test_case in range(12):
    ax = axes[test_case // 3, test_case % 3]

    # Plot setup
    setup = importlib.import_module(f'{model}.config{test_case}')
    mesh = Mesh(f'{os.path.abspath(os.path.dirname(__file__))}/{model}/meshes/{test_case}.msh')
    tags = setup.parameters.turbine_ids
    P0 = FunctionSpace(mesh, 'DG', 0)
    footprints = assemble(sum(TestFunction(P0)*dx(tag) for tag in tags))
    footprints.interpolate(conditional(footprints > 0, 0, 1))
    triplot(mesh, axes=ax, boundary_kw={'color': 'dodgerblue'}, interior_kw={'edgecolor': 'w'})
    tricontourf(footprints, axes=ax, cmap='Blues', levels=[0, 1])

    # Adjust axes
    W = assemble(Constant(1.0, domain=mesh)*ds(1))
    L = 0.5*assemble(Constant(1.0, domain=mesh)*ds(3))  # NOTE: both top and bottom are tagged as 3
    dL = 0.5*(xmax-L)
    dW = 0.5*(ymax-W)
    ax.axis(False)
    ax.set_xlim([xmin - dL - eps, xmax - dL + eps])
    ax.set_ylim([ymin - dW - eps, ymax - dW + eps])

    # Annotate with viscosity coefficient and bathymetry
    nu = setup.parameters.viscosity.values()[0]
    b = setup.parameters.depth
    ax.annotate(r'$\nu$' + f' = {nu:.3f}', xy=(0.65*L, 0.85*W), color='darkgrey')
    ax.annotate(r'$b$' + f' = {b:.2f}', xy=(0.65*L, 0.7*W), color='darkgrey')
plt.savefig(f'{model}/plots/test_cases.pdf')
