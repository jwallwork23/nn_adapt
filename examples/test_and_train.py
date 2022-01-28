from nn_adapt.ann import *

import argparse
import importlib
from sklearn import model_selection
from time import perf_counter
from torch.optim.lr_scheduler import StepLR


# Configuration
parser = argparse.ArgumentParser(prog='test_and_train.py')
parser.add_argument('model', help='The equation set being solved')
parser.add_argument('num_training_cases', help='The configuration file number')
parser.add_argument('-adaptation_steps', help='Number of adaptation steps to learn from (default 4)')
parser.add_argument('-lr', help='Starting learning rate, i.e. step length (default 2.0e-03)')
parser.add_argument('-lr_adapt_num_steps', help='Number of steps between learning rate adapts (default 1000)')
parser.add_argument('-lr_adapt_factor', help='Factor by which to reduce the learning rate (default 0.8)')
parser.add_argument('-num_epochs', help='The number of iterations (default 1000)')
parser.add_argument('-preproc', help='Function for preprocessing data (default "arctan")')
parser.add_argument('-batch_size', help='Number of data points per training iteration (default 100)')
parser.add_argument('-test_batch_size', help='Number of data points per validation iteration (default 100)')
parser.add_argument('-test_size', help='Proportion of data used for validation (default 0.3)')
parser.add_argument('-seed', help='Seed for random number generator (default 42)')
parsed_args = parser.parse_args()
model = parsed_args.model
num_training_cases = int(parsed_args.num_training_cases)
adaptation_steps = int(parsed_args.adaptation_steps or 4)
lr = float(parsed_args.lr or 2.0e-03)
assert lr > 0.0
lr_adapt_num_steps = int(parsed_args.lr_adapt_num_steps or 200)
assert lr_adapt_num_steps > 0
lr_adapt_factor = float(parsed_args.lr_adapt_factor or 0.8)
assert 0.0 < lr_adapt_factor < 1.0
num_epochs = int(parsed_args.num_epochs or 1000)
assert num_epochs > 0
preproc = parsed_args.preproc or 'arctan'
batch_size = int(parsed_args.batch_size or 100)
assert batch_size > 0
test_batch_size = int(parsed_args.test_batch_size or 100)
assert test_batch_size > 0
test_size = float(parsed_args.test_size or 0.3)
assert 0.0 < test_size < 1.0
seed = int(parsed_args.seed or 42)
assert seed > 0

# Load data
concat = lambda a, b: b if a is None else np.concatenate((a, b), axis=0)
features = None
targets = None
errors = None
for step in range(adaptation_steps):
    for approach in ('isotropic', 'anisotropic'):
        if step == 0 and approach == 'anisotropic':
            continue
        for test_case in range(num_training_cases):
            features = concat(features, np.load(f'{model}/data/features{test_case}_GO{approach}_{step}.npy'))
            targets = concat(targets, np.load(f'{model}/data/targets{test_case}_GO{approach}_{step}.npy'))
print(f'Total number of features: {len(features.flatten())}')
print(f'Total number of targets: {len(targets)}')
features = torch.from_numpy(preprocess_features(features, preproc=preproc)).type(torch.float32)
targets = torch.from_numpy(targets).type(torch.float32)

# Get train and validation datasets
xtrain, xval, ytrain, yval = model_selection.train_test_split(features, targets, test_size=0.3, random_state=42)
train_data = torch.utils.data.TensorDataset(torch.Tensor(xtrain), torch.Tensor(ytrain))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
validate_data = torch.utils.data.TensorDataset(torch.Tensor(xval), torch.Tensor(yval))
validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=test_batch_size, shuffle=False, num_workers=0)

# Setup model
layout = importlib.import_module(f'{model}.network').NetLayout()
nn = SimpleNet(layout).to(device)
optimizer = torch.optim.Adam(nn.parameters(), lr=lr)
scheduler = StepLR(optimizer, lr_adapt_num_steps, gamma=lr_adapt_factor)
criterion = Loss()
print(f"Model parameters are{'' if all(p.is_cuda for p in nn.parameters()) else ' not'} using GPU cores.")

# Train
train_losses = []
validation_losses = []
adapt_steps = []
set_seed(seed)
for epoch in range(num_epochs):

    # Training step
    start_time = perf_counter()
    train = propagate(train_loader, nn, criterion, optimizer)
    mid_time = perf_counter()
    train_time = mid_time - start_time

    # Validation step
    val = propagate(validate_loader, nn, criterion)
    validation_time = perf_counter() - mid_time

    # Adapt learning rate
    scheduler.step()
    if epoch % lr_adapt_num_steps == 0:
        adapt_steps.append(epoch)
        np.save(f'{model}/data/adapt_steps', adapt_steps)

    # Stash progreess
    print(f"Epoch {epoch:4d}/{num_epochs:d}"
          f"  avg loss: {train:.4e} / {val:.4e}"
          f"  wallclock: {train_time:.2f}s / {validation_time:.2f}s")
    train_losses.append(train)
    validation_losses.append(val)
    np.save(f'{model}/data/train_losses', train_losses)
    np.save(f'{model}/data/validation_losses', validation_losses)
    torch.save(nn.state_dict(), f'{model}/model.pt')
