"""
Plot the training and validation loss curves for a network
trained on a particular ``model``.
"""
from nn_adapt.plotting import *

import argparse
import numpy as np


# Parse model
parser = argparse.ArgumentParser(prog="plot_progress.py")
parser.add_argument("model", help="The equation set being solved")
parser.add_argument("-num_epochs", help="The number of iterations (default 1000)")
parser.add_argument("-git_sha", help="Git commit sha (defaults to current)")
parsed_args = parser.parse_args()
model = parsed_args.model
num_epochs = int(parsed_args.num_epochs or 1000)
assert num_epochs > 0
sha = parsed_args.git_sha
if sha is None:
    import git

    sha = git.Repo(search_parent_directories=True).head.object.hexsha

# Load data
train_losses = np.load(f"{model}/data/train_losses_{sha}.npy")
validation_losses = np.load(f"{model}/data/validation_losses_{sha}.npy")
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
