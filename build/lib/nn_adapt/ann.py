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
    adaptation with a single hidden layer.
    """

    def __init__(self, layout):
        """
        :arg layout: class instance inherited from
            :class:`NetLayoutBase`, with numbers of
            inputs, hidden neurons and outputs
            specified.
        """
        super().__init__()

        # Define layers
        self.linear1 = nn.Linear(layout.num_inputs, layout.num_hidden_neurons)
        self.linear2 = nn.Linear(layout.num_hidden_neurons, 1)

        # Define activation functions
        self.activate1 = nn.Sigmoid()

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

    return cumulative_loss / num_batches


def preprocess_features(features, preproc="none"):
    """
    Pre-process features so that they are
    similarly scaled.

    :arg features: the list of feature arrays
    :kwarg preproc: preprocessor function
    """
    if preproc == "none":
        return features
    if preproc == "arctan":
        f = np.arctan
    elif preproc == "tanh":
        f = np.tanh
    elif preproc == "logabs":
        f = lambda x: np.ln(np.abs(x))
    else:
        raise ValueError(f'Preprocessor "{preproc}" not recognised.')
    for i, feature in features.items():
        features[i] = f(feature.flatten()).reshape(*feature.shape)
    return features


def collect_features(feature_dict, preproc="none"):
    """
    Given a dictionary of feature arrays, stack their
    data appropriately to be fed into a neural network.

    :arg feature_dict: dictionary containing feature data
    :kwarg preproc: preprocessor function
    """

    # Pre-process, if requested
    if preproc != "none":
        feature_dict = preprocess_features(feature_dict, preproc=preproc)

    # Stack appropriately
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
