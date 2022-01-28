from firedrake import *
from nn_adapt.plotting import *

import argparse
import importlib


# Parse model
parser = argparse.ArgumentParser(prog='plot_config.py')
parser.add_argument('model', help='The equation set being solved')
parser.add_argument('-num_columns', help='The number of columns in the plot (default 4)')
parser.add_argument('-num_rows', help='The number of rows in the plot (default 4)')
parsed_args = parser.parse_args()
model = parsed_args.model
ncols = int(parsed_args.num_columns or 4)
nrows = int(parsed_args.num_rows or 4)
N = ncols*nrows

# Plot all configurations
fig, axes = plt.subplots(ncols=nrows, nrows=ncols, figsize=(3*ncols, 1.5*nrows))
for test_case in range(N):
    ax = axes[test_case // ncols, test_case % nrows]

    # Plot setup
    setup = importlib.import_module(f'{model}.config')
    setup.initialise(test_case+1)
    mesh = Mesh(f'{model}/meshes/{test_case+1}.msh')
    importlib.import_module(f'{model}.plotting').plot_config(setup, mesh, ax)
plt.tight_layout()
plt.savefig(f'{model}/plots/training_config.pdf')
