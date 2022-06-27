"""
Plot the sensitivities of a network trained on a
particular ``model`` to its input parameters.
"""
from nn_adapt.parse import argparse, positive_int
from nn_adapt.plotting import *

import git
import importlib
import numpy as np


# Parse model
parser = argparse.ArgumentParser(
    prog="plot_importance.py",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "model",
    help="The model",
    type=str,
    choices=["turbine"],
)
parser.add_argument(
    "num_training_cases",
    help="The number of training cases",
    type=positive_int,
)
parser.add_argument(
    "-a",
    "--approaches",
    nargs="+",
    help="Adaptive approaches to consider",
    choices=["isotropic", "anisotropic"],
    default=["anisotropic"],
)
parser.add_argument(
    "--adaptation_steps",
    help="Steps to learn from",
    type=positive_int,
    default=4,
)
parser.add_argument(
    "--preproc",
    help="Data preprocess function",
    type=str,
    choices=["none", "arctan", "tanh", "logabs"],
    default="arctan",
)
parser.add_argument(
    "--tag",
    help="Model tag (defaults to current git commit sha)",
    default=None,
)
parsed_args = parser.parse_args()
model = parsed_args.model
preproc = parsed_args.preproc
tag = parsed_args.tag or git.Repo(search_parent_directories=True).head.object.hexsha

# Separate sensitivity information by variable
data = np.load(f"{model}/data/sensitivities_{tag}.npy")
layout = importlib.import_module(f"{model}.network").NetLayout()
p = importlib.import_module(f"{model}.plotting")
sensitivities = p.process_sensitivities(data, layout)

# Plot increases as a stacked bar chart
colours = ("b", "C0", "deepskyblue", "mediumturquoise", "mediumseagreen", "0.3")
deriv = ("", "_x", "_y", "_{xx}", "_{xy}", "_{yy}")
N = len(sensitivities.keys())
bottom = np.zeros(N)
fig, axes = plt.subplots(figsize=(1.5 * N, 4))
for i, colour in enumerate(colours):
    arr = np.array([S[i] for S in sensitivities.values()])
    label = r"$f%s(\mathbf x_K)$" % deriv[i]
    axes.bar(sensitivities.keys(), arr, bottom=bottom, color=colour, label=label)
    bottom += arr
xlim = axes.get_xlim()
axes.set_xlabel("Input parameters")
axes.set_ylabel("Network sensitivity")
axes.legend(ncol=2)
axes.grid(True)
plt.tight_layout()
plt.savefig(f"{model}/plots/importance_{tag}.pdf")
