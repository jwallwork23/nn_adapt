from nn_adapt.plotting import *

import argparse
import numpy as np


# Parse model
parser = argparse.ArgumentParser(prog='plot_progress.py')
parser.add_argument('model', help='The equation set being solved')
args = parser.parse_args()
model = args.model
assert model in ['turbine']

# Load data
train_losses = np.load(f'{model}/data/train_losses.npy')
validation_losses = np.load(f'{model}/data/validation_losses.npy')
epochs = np.arange(len(train_losses)) + 1

# Plot training losses
fig, axes = plt.subplots()
axes.semilogx(epochs, train_losses, label='Training', color='deepskyblue')
axes.set_xlabel('Number of epochs')
axes.set_ylabel('Average loss')
axes.legend()
axes.grid(True)
axes.set_xlim([1.0e+00, 1.0e+03])
plt.tight_layout()
plt.savefig(f'{model}/plots/training_losses.pdf')

# Plot validation losses
fig, axes = plt.subplots()
axes.semilogx(epochs, validation_losses, label='Validation', color='deepskyblue')
axes.set_xlabel('Number of epochs')
axes.set_ylabel('Average loss')
axes.legend()
axes.grid(True)
axes.set_xlim([1.0e+00, 1.0e+03])
plt.tight_layout()
plt.savefig(f'{model}/plots/validation_losses.pdf')

# Plot both
fig, axes = plt.subplots()
axes.semilogx(epochs, train_losses, label='Training', color='deepskyblue')
axes.semilogx(epochs, validation_losses, label='Validation', color='darkgreen')
axes.set_xlabel('Number of epochs')
axes.set_ylabel('Average loss')
axes.legend()
axes.grid(True)
axes.set_xlim([1.0e+00, 1.0e+03])
plt.tight_layout()
plt.savefig(f'{model}/plots/losses.pdf')
