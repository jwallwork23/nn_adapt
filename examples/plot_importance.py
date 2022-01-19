from nn_adapt.ann import *
from plotting import *

import argparse
import numpy as np


# Parse model
parser = argparse.ArgumentParser(prog='test_importance.py')
parser.add_argument('model', help='The equation set being solved')
parser.add_argument('test_case', help='The configuration file number')
parsed_args = parser.parse_args()
model = parsed_args.model
assert model in ['stokes', 'turbine']
test_case = int(parsed_args.test_case)
assert test_case in list(range(12))

# Number of perturbations to the input parameters
N = 100


def Loss():
    """
    Custom loss function.

    Needed when there is only one output value.
    """
    def mse(tens1, tens2):
        return torch.nn.MSELoss(reduction='mean')(tens1, tens2.reshape(*tens1.shape))
    return mse


# Load the model
nn = SimpleNet().to(device)
nn.load_state_dict(torch.load(f'{model}/model.pt'))
nn.eval()
loss = Loss()

# Load some data
features = torch.from_numpy(np.load(f'{model}/data/features{test_case}_GOanisotropic_0.npy')).type(torch.float32)
expected = torch.from_numpy(np.load(f'{model}/data/targets{test_case}_GOanisotropic_0.npy')).type(torch.float32)

# Evaluate the loss
before = loss(expected, nn.forward(features))

# Perturb the desired input parameter and evaluate the loss again
increase = []
for i in range(29):
    inc = 0
    for k in range(N):
        eps = np.random.rand(len(features))
        features[:, i] += eps
        after = loss(expected, nn.forward(features))
        features[:, i] -= eps
        inc += abs(after - before)/before*100
    inc /= N
    print(f'Input {i:2d}:  before {before:6.2f}  after {after:6.2f}  increase {inc:6.2f}%')
    increase.append(inc.detach().numpy())

# Plot increases as a bar chart
fig, axes = plt.subplots()
colours = ['0'] + 4*['0.2'] + 12*['0.4'] + 12*['0.6']
axes.bar(list(range(1)), increase[:1], color='0', label='Physics')
axes.bar(list(range(1, 5)), increase[1:5], color='0.2', label='Mesh')
axes.bar(list(range(5, 17)), increase[5:17], color='0.4', label='Forward')
axes.bar(list(range(17, 29)), increase[17:], color='0.6', label='Adjoint')
axes.set_xticks([])
axes.set_xlabel('Input parameters')
axes.set_ylabel('Network sensitivity')
axes.legend(loc='upper left')
plt.tight_layout()
plt.savefig(f'{model}/plots/importance.pdf')
