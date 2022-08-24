"""
Compute the sensitivities of a network trained on a
particular ``model`` to its input parameters.
"""
from nn_adapt.ann import *
from nn_adapt.parse import argparse, positive_int
from nn_adapt.plotting import *

import git
import importlib
import numpy as np


# Parse model
parser = argparse.ArgumentParser(
    prog="compute_importance.py",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "model",
    help="The model",
    type=str,
    choices=["steady_turbine", "pyroteus_burgers"],
)
parser.add_argument(
    "num_training_cases",
    help="The number of training cases",
    type=positive_int,
)
parser.add_argument(
    "-a",
    "--approaches",
    nargs="+",
    help="Adaptive approaches to consider",
    choices=["isotropic", "anisotropic"],
    default=["anisotropic"],
)
parser.add_argument(
    "--adaptation_steps",
    help="Steps to learn from",
    type=positive_int,
    default=3,
)
parser.add_argument(
    "--preproc",
    help="Data preprocess function",
    type=str,
    choices=["none", "arctan", "tanh", "logabs"],
    default="arctan",
)
parser.add_argument(
    "--tag",
    help="Model tag (defaults to current git commit sha)",
    default=git.Repo(search_parent_directories=True).head.object.hexsha,
)
parsed_args = parser.parse_args()
model = parsed_args.model
preproc = parsed_args.preproc
tag = parsed_args.tag

# Load the model
layout = importlib.import_module(f"{model}.network").NetLayout()
nn = SingleLayerFCNN(layout, preproc=preproc).to(device)
nn.load_state_dict(torch.load(f"{model}/model_{tag}.pt"))
nn.eval()
loss_fn = Loss()

# Compute (averaged) sensitivities of the network to the inputs
dJdm = torch.zeros(layout.num_inputs)
data_dir = f"{model}/data"
approaches = parsed_args.approaches
values = np.zeros((0, layout.num_inputs))
for step in range(parsed_args.adaptation_steps):
    for approach in approaches:
        for test_case in range(1, parsed_args.num_training_cases + 1):
            if test_case == 1 and approach != approaches[0]:
                continue
            suffix = f"{test_case}_GO{approach}_{step}"

            # Load some data and mark inputs as independent
            data = {
                key: np.load(f"{data_dir}/feature_{key}_{suffix}.npy")
                for key in layout.inputs
            }
            features = collect_features(data, layout)
            values = np.vstack((values, features))
            features = torch.from_numpy(features).type(torch.float32)
            features.requires_grad_(True)

            # Run the model and sum the outputs
            out = nn(features).sum(axis=0)

            # Backpropagate to get the gradient of the outputs w.r.t. the inputs
            out.backward()
            dJdm += features.grad.mean(axis=0)

# Compute representative values for each parameter
dm = np.abs(np.mean(values, axis=0))

# Multiply by the variability
sensitivity = dJdm.abs().detach().numpy() * dm
np.save(f"{model}/data/sensitivities_{tag}.npy", sensitivity)
