"""
Plot the training and validation loss curves for a network
trained on a particular ``model``.
"""
from nn_adapt.parse import argparse, positive_int
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
    choices=["turbine"],
)
parser.add_argument(
    "--num_epochs",
    help="The number of iterations",
    type=positive_int,
    default=1000,
)
parser.add_argument(
    "--tag",
    help="Model tag (defaults to current git commit sha)",
    default=None,
)
parsed_args = parser.parse_args()
model = parsed_args.model
num_epochs = parsed_args.num_epochs
tag = parsed_args.tag or git.Repo(search_parent_directories=True).head.object.hexsha

# Load data
train_losses = np.load(f"{model}/data/train_losses_{tag}.npy")
validation_losses = np.load(f"{model}/data/validation_losses_{tag}.npy")
epochs = np.arange(len(train_losses)) + 1

# Plot training losses
fig, axes = plt.subplots()
axes.loglog(epochs, train_losses, label="Training", color="deepskyblue")
axes.set_xlabel("Number of epochs")
axes.set_ylabel("Average loss")
axes.legend()
axes.grid(True)
axes.set_xlim([1, num_epochs])
plt.tight_layout()
plt.savefig(f"{model}/plots/training_losses.pdf")

# Plot validation losses
fig, axes = plt.subplots()
axes.loglog(epochs, validation_losses, label="Validation", color="deepskyblue")
axes.set_xlabel("Number of epochs")
axes.set_ylabel("Average loss")
axes.legend()
axes.grid(True)
axes.set_xlim([1, num_epochs])
plt.tight_layout()
plt.savefig(f"{model}/plots/validation_losses.pdf")

# Plot both
fig, axes = plt.subplots()
axes.loglog(epochs, train_losses, label="Training", color="deepskyblue")
axes.loglog(epochs, validation_losses, label="Validation", color="darkgreen")
axes.set_xlabel("Number of epochs")
axes.set_ylabel("Average loss")
axes.legend()
axes.grid(True)
axes.set_xlim([1, num_epochs])
plt.tight_layout()
plt.savefig(f"{model}/plots/losses.pdf")
