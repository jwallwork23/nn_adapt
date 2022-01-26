from nn_adapt.ann import *
from nn_adapt.plotting import *

import argparse
import numpy as np


# Hard-coded parameters
num_test_cases = 16
adaptation_steps = 4

# Parse model
parser = argparse.ArgumentParser(prog='test_importance.py')
parser.add_argument('model', help='The equation set being solved')
parser.add_argument('-preproc', help='Function for preprocessing data (default "arctan")')
parsed_args = parser.parse_args()
model = parsed_args.model
assert model in ['turbine']
preproc = parsed_args.preproc or 'arctan'

# Load the model
nn = SimpleNet().to(device)
nn.load_state_dict(torch.load(f'{model}/model.pt'))
nn.eval()
loss_fn = Loss()

# Category metadata
categories = {
    'Physics': {'num': 3, 'colour': 'C0'},
    'Mesh': {'num': 3, 'colour': 'deepskyblue'},
    'Forward': {'num': 12, 'colour': 'mediumturquoise'},
    'Adjoint': {'num': 12, 'colour': 'mediumseagreen'},
}
num_inputs = sum([md['num'] for label, md in categories.items()])

# Compute (averaged) sensitivities of the network to the inputs
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
i = 0
for label, md in categories.items():
    k = md['num']
    axes.bar(np.arange(i, i+k), sensitivity[i:i+k], color=md['colour'], label=label)
    i += k
xlim = axes.get_xlim()
axes.set_xlim([xlim[0]+1.25, xlim[1]-1.25])
axes.set_xticks([])
axes.set_yticks([])
axes.set_xlabel('Input parameters')
axes.set_ylabel('Network sensitivity')
axes.legend(loc='best')
axes.grid(True)
plt.tight_layout()
plt.savefig(f'{model}/plots/importance.pdf')
