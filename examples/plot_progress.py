"""
Plot the training and validation loss curves for a network
trained on a particular ``model``.
"""
from nn_adapt.parse import argparse
from nn_adapt.plotting import *

import git
import numpy as np


# Parse model
parser = argparse.ArgumentParser(
    prog="plot_progress.py",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "model",
    help="The model",
    type=str,
    choices=["steady_turbine", "pyroteus_burgers"],
)
parser.add_argument(
    "--tag",
    help="Model tag (defaults to current git commit sha)",
    default=None,
)
parsed_args = parser.parse_args()
model = parsed_args.model
tag = parsed_args.tag or git.Repo(search_parent_directories=True).head.object.hexsha

# Load data
train_losses = np.load(f"{model}/data/train_losses_{tag}.npy")
validation_losses = np.load(f"{model}/data/validation_losses_{tag}.npy")
epochs = np.arange(len(train_losses)) + 1

# Plot losses
fig, axes = plt.subplots()
kw = dict(linewidth=0.5)
axes.loglog(epochs, train_losses, label="Training", color="deepskyblue", **kw)
axes.loglog(epochs, validation_losses, label="Validation", color="darkgreen", **kw)
axes.set_xlabel("Number of epochs")
axes.set_ylabel("Average loss")
axes.legend()
axes.grid(True)
axes.set_xlim([1, epochs[-1]])
plt.tight_layout()
plt.savefig(f"{model}/plots/losses_{tag}.pdf")
