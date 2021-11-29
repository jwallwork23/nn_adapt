import random
import numpy as np
import torch
from torch import nn
from time import perf_counter


# Set device
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    dev = torch.cuda.current_device()
    print(f"Cuda installed. Running on GPU {dev}.")
    device = torch.device(f"cuda:{dev}")
    torch.backends.cudnn.benchmark = True
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

    # Use the inbuilt cudnn auto-tuner to find the fastest convolution algorithm
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    return True


class SimpleNet(nn.Module):
    """
    Fully Connected Neural Network (FCNN)
    for goal-oriented metric-based mesh
    adaptation.

    Input layer:
    ============
        [9 forward DoFs per element]
          + [9 adjoint DoFs per element]
          + [element orientation]
          + [element shape]
          + [element size]
          + [mesh Reynolds number]
          + [boundary element?]
          + [error indicator on coarse mesh]
          = 24

    Hidden layer:
    =============
        60 neurons

    Output layer:
    =============
        [1 error indicator value]
    """
    def __init__(self, num_inputs=24, num_outputs=1, num_hidden_neurons=60):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden_neurons)
        self.activate1 = nn.Sigmoid()
        self.linear2 = nn.Linear(num_hidden_neurons, num_outputs)

    def forward(self, x):
        z1 = self.linear1(x)
        a1 = self.activate1(z1)
        z2 = self.linear2(a1)
        return z2


def train(data_loader, model, loss_fn, optimizer):
    """
    Train the neural network.

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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return cumulative_loss


def validate(data_loader, model, loss_fn, epoch, num_epochs, timestamp):
    """
    Validate the neural network.

    :arg data_loader: PyTorch :class:`DataLoader` instance
    :arg model: PyTorch :class:`Module` instance
    :arg loss_fn: PyTorch loss function instance
    :arg epoch: the current epoch
    :arg num_epochs: the total number of epochs
    :arg timestamp: timestamp for the start of this epoch
    """
    num_batches = len(data_loader)
    cumulative_loss = 0

    with torch.no_grad():
        for x, y in data_loader:
            prediction = model(x.to(device))
            loss = loss_fn(prediction, y.to(device))
            cumulative_loss += loss.item()

    print(f"Epoch {epoch:4d}/{num_epochs:d}"
          f"  avg loss: {cumulative_loss/num_batches:.4e}"
          f"  wallclock {perf_counter() - timestamp:.1f}s")
    return cumulative_loss
