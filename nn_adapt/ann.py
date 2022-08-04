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


class SingleLayerFCNN(nn.Module):
    """
    Fully Connected Neural Network (FCNN)
    for goal-oriented metric-based mesh
    adaptation with single hidden layer.
    """

    def __init__(self, layout, preproc="arctan"):
        """
        :arg layout: class instance inherited from
            :class:`NetLayoutBase`, with numbers of
            inputs, hidden neurons and outputs
            specified.
        :kwarg preproc: pre-processing function to
            apply to the input data
        """
        super().__init__()
        
        # Define preprocessing function
        if preproc == "none":
            self.preproc1 = lambda x: x
        if preproc == "arctan":
            self.preproc1 = torch.arctan
        elif preproc == "tanh":
            self.preproc1 = torch.tanh
        elif preproc == "logabs":
            self.preproc1 = lambda x: torch.log(torch.abs(x))
        else:
            raise ValueError(f'Preprocessor "{preproc}" not recognised.')

        # Define layers
        self.linear1 = nn.Linear(layout.num_inputs, layout.num_hidden_neurons)

        self.linear2 = nn.Linear(layout.num_hidden_neurons, layout.num_hidden_neurons)
        self.linear3 = nn.Linear(layout.num_hidden_neurons, layout.num_hidden_neurons)
        self.linear4 = nn.Linear(layout.num_hidden_neurons, layout.num_hidden_neurons)
        self.linear5 = nn.Linear(layout.num_hidden_neurons, layout.num_hidden_neurons)
        self.linear6 = nn.Linear(layout.num_hidden_neurons, layout.num_hidden_neurons)
        self.linear7 = nn.Linear(layout.num_hidden_neurons, layout.num_hidden_neurons)
        self.linear8 = nn.Linear(layout.num_hidden_neurons, layout.num_hidden_neurons)
        self.linear9 = nn.Linear(layout.num_hidden_neurons, layout.num_hidden_neurons)
        self.linear10 = nn.Linear(layout.num_hidden_neurons, layout.num_hidden_neurons)
        
        self.linear11 = nn.Linear(layout.num_hidden_neurons, 1)

        # Define activation functions

        self.activate = nn.Sigmoid() # suppose we don't change activation function for now

    def forward(self, x):
        p = self.preproc1(x)

        z1 = self.linear1(p)
        a1 = self.activate(z1)
        
        z2 = self.linear2(a1)
        a2 = self.activate(z2)

        z3 = self.linear3(a2)
        a3 = self.activate(z3)

        z4 = self.linear4(a3)
        a4 = self.activate(z4)

        z5 = self.linear5(a4)
        a5 = self.activate(z5)

        z6 = self.linear6(a5)
        a6 = self.activate(z6)

        z7 = self.linear7(a6)
        a7 = self.activate(z7)

        z8 = self.linear8(a7)
        a8 = self.activate(z8)

        z9 = self.linear9(a8)
        a9 = self.activate(z9)

        z10 = self.linear10(a9)
        a10 = self.activate(z10)

        z11 = self.linear11(a10)
        return z11


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

    return cumulative_loss / num_batches


def collect_features(feature_dict):
    """
    Given a dictionary of feature arrays, stack their
    data appropriately to be fed into a neural network.

    :arg feature_dict: dictionary containing feature data
    """
    dofs = [feature for key, feature in feature_dict.items() if "dofs" in key]
    nodofs = [feature for key, feature in feature_dict.items() if "dofs" not in key]
    return np.hstack((np.vstack(nodofs).transpose(), np.hstack(dofs)))


def Loss():
    """
    Custom loss function.

    Needed when there is only one output value.
    """

    def mse(output, target):
        target = target.reshape(*output.shape)
        return torch.nn.MSELoss(reduction="sum")(output, target)

    return mse
