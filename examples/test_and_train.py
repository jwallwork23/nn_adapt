import argparse
import os
from sklearn import model_selection
import matplotlib.pyplot as plt
from nn_adapt.ann import *


# Configuration
parser = argparse.ArgumentParser(prog='test_and_train.py')
parser.add_argument('model', help='The equation set being solved')
parser.add_argument('-learning_rate', help='The step length (default 5.0e-05)')
parser.add_argument('-num_epochs', help='The number of iterations (default 4000)')
parser.add_argument('-preproc', help='Function for preprocessing data (default "arctan")')
parser.add_argument('-batch_size')
parser.add_argument('-test_batch_size')
args = parser.parse_args()
model = args.model
assert model in ['stokes']
lr = float(args.learning_rate or 5.0e-05)
num_epochs = int(args.num_epochs or 4000)
preproc = args.preproc or 'arctan'
batch_size = int(args.batch_size or 32)
test_batch_size = int(args.test_batch_size or 100)
input_dir = os.path.join(model, 'data')
plot_dir = os.path.join(model, 'plots')

# Setup empty arrays
features = np.array([]).reshape(0, 58)
targets = np.array([]).reshape(0, 3)
errors = np.array([])

# Load data
load = lambda name: np.load(os.path.join(input_dir, f'{name}.npy'))
for run in ['GO0', 'GO1', 'GO2', 'GO3']:
    for i in [0, 1, 2, 3]:
        features = np.concatenate((features, load(f'features{i}_{run}')), axis=0)
        targets = np.concatenate((targets, load(f'targets{i}_{run}')), axis=0)
        errors = np.concatenate((errors, load(f'indicator{i}_{run}')), axis=0)

# Pre-process features
shape = features.shape
if preproc == 'arctan':
    f = np.arctan
elif preproc == 'tanh':
    f = np.tanh
elif preproc == 'logabs':
    f = lambda x: np.ln(np.abs(x))
elif preproc != 'none':
    raise ValueError(f'Preprocessor "{preproc}" not recognised.')
if preproc != 'none':
    features = f(features.reshape(1, shape[0]*shape[1])).reshape(*shape)

# Get train and validation datasets
xtrain, xval, ytrain, yval = model_selection.train_test_split(features, targets, test_size=0.2, random_state=42)
train_data = torch.utils.data.TensorDataset(torch.Tensor(xtrain), torch.Tensor(ytrain))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
validate_data = torch.utils.data.TensorDataset(torch.Tensor(xval), torch.Tensor(yval))
validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=test_batch_size, shuffle=False, num_workers=0)

# Setup model
nn = SimpleNet().to(device)
optimizer = torch.optim.Adam(nn.parameters(), lr=lr)
# criterion = torch.nn.MSELoss(reduction='sum')
criterion = torch.nn.MSELoss()
print(f"Model parameters are{'' if all(p.is_cuda for p in nn.parameters()) else ' not'} using GPU cores.")

# Train
epochs = []
train_losses = []
validation_losses = []
set_seed(42)
epochs = range(num_epochs)
for epoch in epochs:
    timestamp = perf_counter()
    train_losses.append(train(train_loader, nn, criterion, optimizer))
    validation_losses.append(validate(validate_loader, nn, criterion, epoch, num_epochs, timestamp))
epochs = list(epochs)

# Plot losses
fig, axes = plt.subplots()
axes.semilogx(epochs, train_losses, label='Training loss')
axes.set_xlabel('Number of epochs')
axes.set_ylabel('Mean squared error loss')
axes.legend()
axes.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'training_losses.pdf'))

# Plot losses
fig, axes = plt.subplots()
axes.semilogx(epochs, validation_losses, label='Validation loss')
axes.set_xlabel('Number of epochs')
axes.set_ylabel('Mean squared error loss')
axes.legend()
axes.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'validation_losses.pdf'))

# Save the model
torch.save(nn.state_dict(), os.path.join(model, 'model.pt'))
