"""
Plot the problem configurations for a given ``model``.
The ``mode`` is chosen from 'train' and 'test'.
"""
from firedrake import *
from nn_adapt.plotting import *

import argparse
import importlib


# Parse model
parser = argparse.ArgumentParser(prog="plot_config.py")
parser.add_argument("model", help="The equation set being solved")
parser.add_argument("mode", help='Choose from "train" and "test"')
parser.add_argument(
    "-num_columns", help="The number of columns in the plot (default 4)"
)
parser.add_argument("-num_rows", help="The number of rows in the plot (default 4)")
parsed_args = parser.parse_args()
model = parsed_args.model
mode = parsed_args.mode
assert mode in ["train", "test"]
setup = importlib.import_module(f"{model}.config")
cases = setup.testing_cases
ncols = int(parsed_args.num_columns or len(cases) if mode == "test" else 4)
nrows = int(parsed_args.num_rows or 1 if mode == "test" else 4)
N = ncols * nrows
if mode == "train":
    cases = range(1, N + 1)
p = importlib.import_module(f"{model}.plotting")

# Plot all configurations
fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(3 * ncols, 1.5 * nrows))
for i, case in enumerate(cases):
    ax = axes[i] if nrows == 1 else axes[i // ncols, i % nrows]
    setup.initialise(case)
    mesh = Mesh(f"{model}/meshes/{case}.msh")
    p.plot_config(setup, mesh, ax)
plt.tight_layout()
plt.savefig(f"{model}/plots/{mode}_config.pdf")
