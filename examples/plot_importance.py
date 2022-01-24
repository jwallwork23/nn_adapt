from nn_adapt.ann import *
from plotting import *

import argparse
import numpy as np


# Hard-coded parameters
num_test_cases = 12
num_inputs = 29
adaptation_steps = 4

# Parse model
parser = argparse.ArgumentParser(prog='test_importance.py')
parser.add_argument('model', help='The equation set being solved')
parser.add_argument('-preproc', help='Function for preprocessing data (default "arctan")')
parsed_args = parser.parse_args()
model = parsed_args.model
assert model in ['stokes', 'turbine']
preproc = parsed_args.preproc or 'arctan'

# Load the model
nn = SimpleNet().to(device)
nn.load_state_dict(torch.load(f'{model}/model.pt'))
nn.eval()
loss_fn = Loss()

sensitivity = torch.zeros(num_inputs)
count = 0
for approach in ['isotropic', 'anisotropic']:
    for step in range(adaptation_steps):
        if step == 0 and approach == 'anisotropic':
            continue
        for test_case in range(num_test_cases):

            # Load some data and mark inputs as independent
            features = preprocess_features(np.load(f'{model}/data/features{test_case}_GO{approach}_{step}.npy'), preproc=preproc)
            features = torch.from_numpy(features).type(torch.float32)
            expected = torch.from_numpy(np.load(f'{model}/data/targets{test_case}_GO{approach}_{step}.npy')).type(torch.float32)
            features.requires_grad_(True)

            # Evaluate the loss
            loss = loss_fn(expected, nn(features))

            # Backpropagate and average to get the sensitivities
            loss.backward()
            sensitivity += features.grad.abs().mean(axis=0)
        count += 1
sensitivity /= num_test_cases*adaptation_steps

# Plot increases as a bar chart
fig, axes = plt.subplots()
axes.bar(list(range(2)), sensitivity[:2], color='C0', label='Physics')
axes.bar(list(range(2, 5)), sensitivity[2:5], color='deepskyblue', label='Mesh')
axes.bar(list(range(5, 17)), sensitivity[5:17], color='mediumturquoise', label='Forward')
axes.bar(list(range(17, 29)), sensitivity[17:], color='mediumseagreen', label='Adjoint')
axes.set_xticks([])
axes.set_xlabel('Input parameters')
axes.set_ylabel('Network sensitivity')
axes.legend(loc='upper left')
plt.tight_layout()
plt.savefig(f'{model}/plots/importance.pdf')
