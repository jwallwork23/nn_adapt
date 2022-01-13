import argparse
import matplotlib.pyplot as plt
import numpy as np


# Parse for test case
parser = argparse.ArgumentParser()
parser.add_argument('model', help='The model')
parser.add_argument('test_case', help='The configuration file number')
parsed_args = parser.parse_args()
model = parsed_args.model
assert model in ['stokes', 'turbine']
test_case = int(parsed_args.test_case)
assert test_case in list(range(12))
approaches = {
    'uniform': {'label': 'Uniform refinement', 'color': 'cornflowerblue', 'marker': 'x', 'linestyle': '-'},
    'GOisotropic': {'label': 'Isotropic goal-oriented adaptation', 'color': 'orange', 'marker': '^', 'linestyle': '-'},
    'GOanisotropic': {'label': 'Anisotropic goal-oriented adaptation', 'color': 'g', 'marker': 'o', 'linestyle': '-'},
    'MLisotropic': {'label': 'Isotropic data-driven adaptation', 'color': 'orange', 'marker': '^', 'linestyle': '--'},
    'MLanisotropic': {'label': 'Anisotropic data-driven adaptation', 'color': 'g', 'marker': 'o', 'linestyle': '--'},
}

# Plot
fig, axes = plt.subplots()
for approach, metadata in approaches.items():
    try:
        dofs = np.load(f'{model}/data/dofs_{approach}_{test_case}.npy')
        qois = np.load(f'{model}/data/qois_{approach}_{test_case}.npy')
    except IOError:
        print(f'Cannot load {approach} data for test case {test_case}')
        continue
    axes.semilogx(dofs, qois, **metadata)
axes.set_xlabel('DoF count')
axes.set_ylabel('Quantity of Interest')
axes.grid(True)
axes.legend()
plt.tight_layout()
plt.savefig(f'{model}/plots/qoi_convergence{test_case}.pdf')
