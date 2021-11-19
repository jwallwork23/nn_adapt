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
        [3 Stokes fields] x [3 unique Hessian entries]
          x [2 forward/adjoint]
          + [element orientation]
          + [element shape]
          + [element size]
          + [mesh Reynolds number]
          + [boundary element?]
          = 23

    Hidden layer:
    =============
        50 neurons

    Output layer:
    =============
        [3 unique metric entries]
    """
    def __init__(self, num_inputs=23, num_outputs=3, num_hidden_neurons=50):
        super(SimpleNet, self).__init__()
        self.linear_1 = nn.Linear(num_inputs, num_hidden_neurons)
        self.activate_1 = nn.Tanh()
        self.linear_2 = nn.Linear(num_hidden_neurons, num_outputs)

    def forward(self, x):
        z1 = self.linear_1(x)
        a1 = self.activate_1(z1)
        z2 = self.linear_2(a1)
        return z2


def train(data_loader, model, loss_fn, optimizer):
    """
    Train the neural network.

    :arg data_loader: PyTorch :class:`DataLoader` instance
    :arg model: PyTorch :class:`Module` instance
    :arg loss_fn: PyTorch loss function instance
    :arg optimizer: PyTorch optimizer instance
    """
    cumulative_loss = 0

    for x, y in data_loader:

        # Compute prediction and loss
        x, y = x.to(device), y.to(device)
        prediction = model(x)
        loss = loss_fn(prediction, y)
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
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            cumulative_loss += loss_fn(prediction, y).item()

    print(f"Epoch {epoch:4d}/{num_epochs}"
          f"  avg loss: {cumulative_loss/num_batches:.4e}"
          f"  wallclock {perf_counter() - timestamp:.1f}s")
    return cumulative_loss
