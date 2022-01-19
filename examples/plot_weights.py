from nn_adapt.ann import *
from plotting import *

import argparse
import torch


# Parse model
parser = argparse.ArgumentParser(prog='plot_weights.py')
parser.add_argument('model', help='The equation set being solved')
args = parser.parse_args()
model = args.model
assert model in ['stokes', 'turbine']

# Load the model
nn = SimpleNet().to(device)
nn.load_state_dict(torch.load(f'{model}/model.pt'))
nn.eval()

# Plot the network
fig, axes = plt.subplots(ncols=2)
im = axes[1].imshow(nn.linear1.weight.data)
eps = 0.53
w = 10
y = 62
axes[1].hlines(y, 0 - eps, 0 + eps, color='0', linewidth=w, label='Physics')
axes[1].hlines(y, 1 - eps, 4 + eps, color='0.2', linewidth=w, label='Mesh')
axes[1].hlines(y, 5 - eps, 16 + eps, color='0.4', linewidth=w, label='Forward solution')
axes[1].hlines(y, 17 - eps, 28 + eps, color='0.6', linewidth=w, label='Adjoint solution')
axes[1].axis(False)
axes[0].axis(False)
cb = fig.colorbar(im, ax=axes[1], anchor=(0.0, 0.5), shrink=0.8)
cb.set_label('Weight')
axes[0].legend(*axes[1].get_legend_handles_labels(), bbox_to_anchor=(1.2, 1.0))
plt.savefig(f'{model}/plots/weights.pdf')
