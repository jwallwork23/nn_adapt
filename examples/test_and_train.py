import argparse
from sklearn import model_selection
from nn_adapt.ann import *


# Configuration
parser = argparse.ArgumentParser(prog='test_and_train.py')
parser.add_argument('model', help='The equation set being solved')
parser.add_argument('-learning_rate', help='The step length (default 2.0e-03)')
parser.add_argument('-num_epochs', help='The number of iterations (default 1000)')
parser.add_argument('-preproc', help='Function for preprocessing data (default "arctan")')
parser.add_argument('-batch_size', help='Number of data points per training iteration (default 1000)')
parser.add_argument('-test_batch_size', help='Number of data points per validation iteration (default 1000)')
parser.add_argument('-test_size', help='Proportion of data used for validation (default 0.3)')
args = parser.parse_args()
model = args.model
assert model in ['stokes', 'turbine']
lr = float(args.learning_rate or 2.0e-03)
num_epochs = int(args.num_epochs or 1000)
preproc = args.preproc or 'arctan'
batch_size = int(args.batch_size or 1000)
assert batch_size > 0
test_batch_size = int(args.test_batch_size or 1000)
assert test_batch_size > 0
test_size = float(args.test_size or 0.3)
assert 0.0 < test_size < 1.0

# Load data
concat = lambda a, b: b if a is None else np.concatenate((a, b), axis=0)
features = None
targets = None
errors = None
for run in range(4):
    for approach in ('isotropic', 'anisotropic'):
        for i in range(9):
            features = concat(features, np.load(f'{model}/data/features{i}_GO{approach}_{run}.npy'))
            targets = concat(targets, np.load(f'{model}/data/targets{i}_GO{approach}_{run}.npy'))
print(f'Total number of features: {len(features.flatten())}')
print(f'Total number of targets: {len(targets)}')

# Pre-process features
shape = features.shape
if preproc == 'arctan':
    f = np.arctan
elif preproc == 'tanh':
    f = np.tanh
elif preproc == 'logabs':
    f = lambda x: np.log(np.abs(x))
elif preproc != 'none':
    raise ValueError(f'Preprocessor "{preproc}" not recognised.')
if preproc != 'none':
    features = f(features.reshape(1, shape[0]*shape[1])).reshape(*shape)

# Get train and validation datasets
xtrain, xval, ytrain, yval = model_selection.train_test_split(features, targets, test_size=0.3, random_state=42)
train_data = torch.utils.data.TensorDataset(torch.Tensor(xtrain), torch.Tensor(ytrain))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
validate_data = torch.utils.data.TensorDataset(torch.Tensor(xval), torch.Tensor(yval))
validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=test_batch_size, shuffle=False, num_workers=0)


def Loss(size_average=None, reduce=None, reduction: str = 'mean'):
    """
    Custom loss function.

    Needed when there is only one output value.
    """
    def mse(tens1, tens2):
        return torch.nn.MSELoss(size_average, reduce, reduction)(tens1, tens2.reshape(*tens1.shape))
    return mse


# Setup model
nn = SimpleNet().to(device)
optimizer = torch.optim.Adam(nn.parameters(), lr=lr)
criterion = Loss()
print(f"Model parameters are{'' if all(p.is_cuda for p in nn.parameters()) else ' not'} using GPU cores.")

# Train
epochs = []
train_losses = []
validation_losses = []
set_seed(42)
for epoch in range(num_epochs):
    timestamp = perf_counter()
    epochs.append(epoch)
    train_losses.append(train(train_loader, nn, criterion, optimizer))
    validation_losses.append(validate(validate_loader, nn, criterion, epoch, num_epochs, timestamp))

    # Stash progreess
    np.save(f'{model}/data/epochs', epochs)
    np.save(f'{model}/data/train_losses', train_losses)
    np.save(f'{model}/data/validation_losses', validation_losses)
    torch.save(nn.state_dict(), f'{model}/model.pt')
