import argparse
from sklearn import model_selection
from nn_adapt.ann import *


# Configuration
parser = argparse.ArgumentParser(prog='test_and_train.py')
parser.add_argument('model', help='The equation set being solved')
parser.add_argument('-learning_rate', help='The step length (default 5.0e-05)')
parser.add_argument('-num_epochs', help='The number of iterations (default 100)')
parser.add_argument('-preproc', help='Function for preprocessing data (default "none")')
parser.add_argument('-batch_size')
parser.add_argument('-test_batch_size')
args = parser.parse_args()
model = args.model
assert model in ['stokes', 'turbine']
lr = float(args.learning_rate or 5.0e-05)
num_epochs = int(args.num_epochs or 100)
preproc = args.preproc or 'none'
batch_size = int(args.batch_size or 32)
test_batch_size = int(args.test_batch_size or 100)

# Load data
concat = lambda a, b: b if a is None else np.concatenate((a, b), axis=0)
features = None
targets = None
errors = None
for run in ['GO0', 'GO1', 'GO2', 'GO3']:
    for i in [0, 1, 2, 3]:
        features = concat(features, np.load(f'{model}/data/features{i}_{run}.npy'))
        targets = concat(targets, np.load(f'{model}/data/targets{i}_{run}.npy'))
        errors = concat(errors, np.load(f'{model}/data/indicator{i}_{run}.npy'))

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


def MSE_Loss(size_average=None, reduce=None, reduction: str = 'mean'):
    """
    Custom mean square error loss.

    Needed when there is only one output value.
    """
    def mse(tens1, tens2):
        return torch.nn.MSELoss(size_average, reduce, reduction)(tens1, tens2.reshape(*tens1.shape))
    return mse


# Setup model
nn = SimpleNet().to(device)
optimizer = torch.optim.Adam(nn.parameters(), lr=lr)
criterion = MSE_Loss()
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
