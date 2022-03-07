"""
Compute and plot the sensitivities of a network
trained on a particular ``model`` to its input parameters.
"""
from nn_adapt.ann import *
from nn_adapt.plotting import *

import argparse
import importlib
import numpy as np


# Parse model
parser = argparse.ArgumentParser(prog="test_importance.py")
parser.add_argument("model", help="The equation set being solved")
parser.add_argument("num_training_cases", help="The number of training cases")
parser.add_argument("-adaptation_steps", help="Steps to learn from (default 4)")
parser.add_argument("-preproc", help="Data preprocess function (default 'arctan')")
parser.add_argument("-git_sha", help="Git commit sha (defaults to current)")
parsed_args = parser.parse_args()
model = parsed_args.model
preproc = parsed_args.preproc or "arctan"
num_training_cases = int(parsed_args.num_training_cases)
assert num_training_cases > 0
adaptation_steps = int(parsed_args.adaptation_steps or 4)
sha = parsed_args.git_sha
if sha is None:
    import git

    sha = git.Repo(search_parent_directories=True).head.object.hexsha

# Load the model
layout = importlib.import_module(f"{model}.network").NetLayout()
nn = SimpleNet(layout).to(device)
nn.load_state_dict(torch.load(f"{model}/model_{sha}.pt"))
nn.eval()
loss_fn = Loss()

# Category metadata
categories = {
    "Estimator": {"num": 1, "colour": "blue"},
    "Physics": {"num": 2, "colour": "C0"},
    "Mesh": {"num": 3, "colour": "deepskyblue"},
    "Forward": {"num": 12, "colour": "mediumturquoise"},
    "Adjoint": {"num": 12, "colour": "mediumseagreen"},
}
num_inputs = sum([md["num"] for label, md in categories.items()])

# Compute (averaged) sensitivities of the network to the inputs
sensitivity = torch.zeros(num_inputs)
count = 0
for step in range(adaptation_steps):
    for test_case in range(1, num_training_cases + 1):

        # Load some data and mark inputs as independent
        features = preprocess_features(
            np.load(f"{model}/data/features{test_case}_GOanisotropic_{step}.npy"),
            preproc=preproc,
        )
        features = torch.from_numpy(features).type(torch.float32)
        expected = torch.from_numpy(
            np.load(f"{model}/data/targets{test_case}_GOanisotropic_{step}.npy")
        ).type(torch.float32)
        features.requires_grad_(True)

        # Evaluate the loss
        loss = loss_fn(expected, nn(features))

        # Backpropagate and average to get the sensitivities
        loss.backward()
        sensitivity += features.grad.abs().mean(axis=0)
    count += 1
sensitivity /= num_training_cases * adaptation_steps

# Plot increases as a bar chart
fig, axes = plt.subplots()
i = 0
for label, md in categories.items():
    j = i + md["num"]
    axes.bar(np.arange(i, j), sensitivity[i:j], color=md["colour"], label=label)
    i = j
xlim = axes.get_xlim()
axes.set_xlim([xlim[0] + 1.25, xlim[1] - 1.25])
axes.set_xticks([])
axes.set_yticks([])
axes.set_xlabel("Input parameters")
axes.set_ylabel("Network sensitivity")
axes.legend(loc="best")
axes.grid(True)
plt.tight_layout()
plt.savefig(f"{model}/plots/importance.pdf")
