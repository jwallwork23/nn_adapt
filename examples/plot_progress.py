from plotting import *

import argparse
import numpy as np


# Parse model
parser = argparse.ArgumentParser(prog='plot_progress.py')
parser.add_argument('model', help='The equation set being solved')
parser.add_argument('-batch_size', help='Number of data points per training iteration (default 1000)')
parser.add_argument('-test_batch_size', help='Number of data points per validation iteration (default 1000)')
parser.add_argument('-test_size', help='Proportion of data used for validation (default 0.3)')
args = parser.parse_args()
model = args.model
assert model in ['stokes', 'turbine']
batch_size = int(args.batch_size or 1000)
assert batch_size > 0
test_batch_size = int(args.test_batch_size or 1000)
assert test_batch_size > 0
test_size = float(args.test_size or 0.3)
assert 0.0 < test_size < 1.0
theta = (1-test_size)*batch_size
theta_test = test_size*test_batch_size

# Load data
epochs = np.load(f'{model}/data/epochs.npy') + 1
train_losses = np.load(f'{model}/data/train_losses.npy')/theta
validation_losses = np.load(f'{model}/data/validation_losses.npy')/theta_test

# Plot training losses
fig, axes = plt.subplots()
axes.semilogx(epochs, train_losses, label='Training')
axes.set_xlabel('Number of epochs')
axes.set_ylabel('Mean squared error loss')
axes.legend()
axes.grid(True)
axes.set_xlim([1.0e+00, 1.0e+03])
plt.tight_layout()
plt.savefig(f'{model}/plots/training_losses.pdf')

# Plot validation losses
fig, axes = plt.subplots()
axes.semilogx(epochs, validation_losses, label='Validation')
axes.set_xlabel('Number of epochs')
axes.set_ylabel('Mean squared error loss')
axes.legend()
axes.grid(True)
axes.set_xlim([1.0e+00, 1.0e+03])
plt.tight_layout()
plt.savefig(f'{model}/plots/validation_losses.pdf')

# Plot both
fig, axes = plt.subplots()
axes.semilogx(epochs, train_losses, label='Training', color='C0')
axes.semilogx(epochs, validation_losses, label='Validation', color='C2')
axes.set_xlabel('Number of epochs')
axes.set_ylabel('Mean squared error loss')
axes.legend()
axes.grid(True)
axes.set_xlim([1.0e+00, 1.0e+03])
plt.tight_layout()
plt.savefig(f'{model}/plots/losses.pdf')
