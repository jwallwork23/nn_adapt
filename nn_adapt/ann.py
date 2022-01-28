"""
Classes and functions related to using neural networks.
"""
import random
import numpy as np
import torch
from torch import nn


# Set device
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    dev = torch.cuda.current_device()
    print(f"Cuda installed. Running on GPU {dev}.")
    device = torch.device(f"cuda:{dev}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
else:
    print("No GPU available.")
    device = torch.device("cpu")


def set_seed(seed):
    """
    Set all random seeds to a fixed value

    :arg seed: the seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleNet(nn.Module):
    """
    Fully Connected Neural Network (FCNN)
    for goal-oriented metric-based mesh
    adaptation.
    """
    def __init__(self, layout):
        """
        :arg layout: class instance inherited from
            :class:`NetLayoutBase`, with numbers of
            inputs, hidden neurons and outputs
            specified.
        """
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(layout.num_inputs, layout.num_hidden_neurons)
        self.activate1 = nn.Sigmoid()
        self.linear2 = nn.Linear(layout.num_hidden_neurons, layout.num_outputs)

    def forward(self, x):
        z1 = self.linear1(x)
        a1 = self.activate1(z1)
        z2 = self.linear2(a1)
        return z2


def propagate(data_loader, model, loss_fn, optimizer=None):
    """
    Propagate data from a :class:`DataLoader` object
    through the neural network.

    If ``optimizer`` is not ``None`` then training is
    performed. Otherwise, validation is performed.

    :arg data_loader: PyTorch :class:`DataLoader` instance
    :arg model: PyTorch :class:`Module` instance
    :arg loss_fn: PyTorch loss function instance
    :arg optimizer: PyTorch optimizer instance
    """
    num_batches = len(data_loader)
    cumulative_loss = 0

    for x, y in data_loader:

        # Compute prediction and loss
        prediction = model(x.to(device))
        loss = loss_fn(prediction, y.to(device))
        cumulative_loss += loss.item()

        # Backpropagation
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return cumulative_loss/num_batches


def preprocess_features(features, preproc='none'):
    """
    Pre-process features so that they are
    similarly scaled.

    :arg features: the array of features
    :kwarg preproc: preprocessor function
    """
    if preproc == 'none':
        return features
    if preproc == 'arctan':
        f = np.arctan
    elif preproc == 'tanh':
        f = np.tanh
    elif preproc == 'logabs':
        f = lambda x: np.ln(np.abs(x))
    else:
        raise ValueError(f'Preprocessor "{preproc}" not recognised.')
    shape = features.shape
    return f(features.reshape(1, shape[0]*shape[1])).reshape(*shape)


def Loss():
    """
    Custom loss function.

    Needed when there is only one output value.
    """
    def mse(tens1, tens2):
        return torch.nn.MSELoss(reduction='mean')(tens1, tens2.reshape(*tens1.shape))
    return mse
