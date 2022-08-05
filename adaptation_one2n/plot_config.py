"""
Plot the problem configurations for a given ``model``.
The ``mode`` is chosen from 'train' and 'test'.
"""
from firedrake import Mesh
from nn_adapt.parse import argparse, positive_int
from nn_adapt.plotting import *

import importlib


# Parse model
parser = argparse.ArgumentParser(
    prog="plot_config.py",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "model",
    help="The model",
    type=str,
    choices=["steady_turbine"],
)
parser.add_argument(
    "mode",
    help="Training or testing?",
    type=str,
    choices=["train", "test"],
)
parser.add_argument(
    "--num_cols",
    help="Number of columns in the plot",
    type=positive_int,
    default=4,
)
parser.add_argument(
    "--num_rows",
    help="Number of rows in the plot",
    type=positive_int,
    default=4,
)
parsed_args = parser.parse_args()
model = parsed_args.model
mode = parsed_args.mode
setup = importlib.import_module(f"{model}.config")
cases = setup.testing_cases
ncols = parsed_args.num_cols
if mode == "test":
    ncols = len(cases)
nrows = parsed_args.num_rows
if mode == "test":
    nrows = 1
N = ncols * nrows
if mode == "train":
    cases = range(1, N + 1)
p = importlib.import_module(f"{model}.plotting")

# Plot all configurations
fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(3 * ncols, 1.5 * nrows))
for i, case in enumerate(cases):
    ax = axes[i] if nrows == 1 else axes[i // ncols, i % nrows]
    setup.initialise(case, discrete=True)
    mesh = Mesh(f"{model}/meshes/{case}.msh")
    p.plot_config(setup, mesh, ax)
plt.tight_layout()
plt.savefig(f"{model}/plots/{mode}_config.pdf")
