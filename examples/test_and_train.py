"""
Train a network on ``num_training_cases`` problem
specifications of a given ``model``.
"""
from nn_adapt.ann import *
from nn_adapt.features import collect_features
from nn_adapt.parse import argparse, bounded_float, nonnegative_int, positive_float, positive_int

import git
import importlib
import numpy as np
import os
from sklearn import model_selection
from time import perf_counter
import torch.optim.lr_scheduler as lr_scheduler


# Configuration
pwd = os.path.abspath(os.path.dirname(__file__))
models = [
    name for name in os.listdir(pwd)
    if os.path.isdir(name) and name not in ("__pycache__", "models")
]
parser = argparse.ArgumentParser(
    prog="test_and_train.py",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-m",
    "--model",
    help="The equation set being solved",
    type=str,
    choices=models,
    default="turbine",
)
parser.add_argument(
    "-n",
    "--num_training_cases",
    help="The number of test cases to train on",
    type=positive_int,
    default=100,
)
parser.add_argument(
    "-a",
    "--approaches",
    nargs="+",
    help="Adaptive approaches to consider",
    choices=["isotropic", "anisotropic"],
    default=["anisotropic"],
)
parser.add_argument(
    "--adaptation_steps",
    help="Steps to learn from",
    type=positive_int,
    default=3,
)
parser.add_argument(
    "--lr",
    help="Initial learning rate",
    type=positive_float,
    default=1.0e-03,
)
parser.add_argument(
    "--lr_adapt_num_steps",
    help="Frequency of learning rate adaptation",
    type=nonnegative_int,
    default=0,
)
parser.add_argument(
    "--lr_adapt_factor",
    help="Learning rate reduction factor",
    type=bounded_float(0, 1),
    default=0.9,
)
parser.add_argument(
    "--num_epochs",
    help="The number of iterations",
    type=positive_int,
    default=2000,
)
parser.add_argument(
    "--patience",
    help="The number of iterations before early stopping",
    type=positive_int,
    default=100,
)
parser.add_argument(
    "--preproc",
    help="Data preprocess function",
    type=str,
    choices=["none", "arctan", "tanh", "logabs"],
    default="arctan",
)
parser.add_argument(
    "--batch_size",
    help="Data points per training iteration",
    type=positive_int,
    default=500,
)
parser.add_argument(
    "--test_batch_size",
    help="Data points per validation iteration",
    type=positive_int,
    default=500,
)
parser.add_argument(
    "--test_size",
    help="Data proportion for validation",
    type=bounded_float(0, 1),
    default=0.3,
)
parser.add_argument(
    "--seed",
    help="Seed for random number generator",
    type=positive_int,
    default=42,
)
parser.add_argument(
    "--tag",
    help="Tag for labelling the model (defaults to current git sha)",
    type=str,
    default=git.Repo(search_parent_directories=True).head.object.hexsha,
)
parsed_args = parser.parse_args()
model = parsed_args.model
approaches = parsed_args.approaches
preproc = parsed_args.preproc
num_epochs = parsed_args.num_epochs
lr = parsed_args.lr
lr_adapt_num_steps = parsed_args.lr_adapt_num_steps
lr_adapt_factor = parsed_args.lr_adapt_factor
patience = parsed_args.patience
seed = parsed_args.seed
tag = parsed_args.tag

# Load network layout
layout = importlib.import_module(f"{model}.network").NetLayout()

# Load data
concat = lambda a, b: b if a is None else np.concatenate((a, b), axis=0)
features = None
targets = None
data_dir = f"{model}/data"
for step in range(parsed_args.adaptation_steps):
    for approach in approaches:
        for test_case in range(1, parsed_args.num_training_cases + 1):
            if test_case == 1 and approach != approaches[0]:
                continue
            suffix = f"{test_case}_GO{approach}_{step}"
            data = {
                key: np.load(f"{data_dir}/feature_{key}_{suffix}.npy")
                for key in layout.inputs
            }
            features = concat(features, collect_features(data, preproc=preproc))
            target = np.load(f"{data_dir}/target_{suffix}.npy")
            targets = concat(targets, target)
print(f"Total number of features: {len(features.flatten())}")
print(f"Total number of targets: {len(targets)}")
features = torch.from_numpy(features).type(torch.float32)
targets = torch.from_numpy(targets).type(torch.float32)

# Get train and validation datasets
xtrain, xval, ytrain, yval = model_selection.train_test_split(
    features, targets, test_size=parsed_args.test_size, random_state=seed
)
train_data = torch.utils.data.TensorDataset(torch.Tensor(xtrain), torch.Tensor(ytrain))
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=parsed_args.batch_size, shuffle=True, num_workers=0
)
validate_data = torch.utils.data.TensorDataset(torch.Tensor(xval), torch.Tensor(yval))
validate_loader = torch.utils.data.DataLoader(
    validate_data, batch_size=parsed_args.test_batch_size, shuffle=False, num_workers=0
)

# Setup model
nn = SimpleNet(layout).to(device)
optimizer = torch.optim.Adam(nn.parameters(), lr=lr)
scheduler1 = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=lr_adapt_factor,
    threshold=1.0e-04,
    patience=50,
    verbose=True,
)
if lr_adapt_num_steps > 0:
    scheduler2 = lr_scheduler.StepLR(
        optimizer,
        lr_adapt_num_steps,
        gamma=lr_adapt_factor
    )
else:
    scheduler2 = None
criterion = Loss()
cuda = all(p.is_cuda for p in nn.parameters())
print(f"Model parameters are{'' if cuda else ' not'} using GPU cores.")

# Train
train_losses, validation_losses, lr_adapt_steps = [], [], []
set_seed(seed)
previous_loss = np.inf
trigger_times = 0
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
    scheduler1.step(val)
    if scheduler2 is not None:
        scheduler2.step()
        if epoch % lr_adapt_num_steps == 0:
            lr_adapt_steps.append(epoch)
            np.save(f"{model}/data/lr_adapt_steps_{tag}", lr_adapt_steps)

    # Stash progress
    print(
        f"Epoch {epoch:4d}/{num_epochs:d}"
        f"  avg loss: {train:.4e} / {val:.4e}"
        f"  wallclock: {train_time:.2f}s / {validation_time:.2f}s"
    )
    train_losses.append(train)
    validation_losses.append(val)
    np.save(f"{model}/data/train_losses_{tag}", train_losses)
    np.save(f"{model}/data/validation_losses_{tag}", validation_losses)
    torch.save(nn.state_dict(), f"{model}/model_{tag}.pt")

    # Test for convergence
    if val > previous_loss:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping")
            break
    else:
        trigger_times = 0
        previous_loss = val
