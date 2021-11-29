import argparse
import matplotlib.pyplot as plt
import numpy as np


# Parse model
parser = argparse.ArgumentParser(prog='plot_progress.py')
parser.add_argument('model', help='The equation set being solved')
args = parser.parse_args()
model = args.model
assert model in ['stokes', 'turbine']

# Load data
epochs = np.load(f'{model}/data/epochs.npy') + 1
train_losses = np.load(f'{model}/data/train_losses.npy')
validation_losses = np.load(f'{model}/data/validation_losses.npy')

# Plot losses
fig, axes = plt.subplots()
axes.semilogx(epochs, train_losses, label='Training loss')
axes.set_xlabel('Number of epochs')
axes.set_ylabel('Mean squared error loss')
axes.legend()
axes.grid(True)
plt.tight_layout()
plt.savefig(f'{model}/plots/training_losses.pdf')

# Plot losses
fig, axes = plt.subplots()
axes.semilogx(epochs, validation_losses, label='Validation loss')
axes.set_xlabel('Number of epochs')
axes.set_ylabel('Mean squared error loss')
axes.legend()
axes.grid(True)
plt.tight_layout()
plt.savefig(f'{model}/plots/validation_losses.pdf')
