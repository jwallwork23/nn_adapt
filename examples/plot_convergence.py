import argparse
import matplotlib.pyplot as plt
import numpy as np


# Parse for test case
parser = argparse.ArgumentParser()
parser.add_argument('model', help='The model')
parser.add_argument('test_case', help='The configuration file number')
parsed_args = parser.parse_args()
model = parsed_args.model
assert model in ['stokes']
test_case = int(parsed_args.test_case)
assert test_case in [0, 1, 2, 3, 4]
labels = ['Uniform refinement', 'Goal-oriented adaptation']

# Plot
fig, axes = plt.subplots()
for approach, label in zip(['uniform', 'go'], labels):
    try:
        dofs = np.load(f'{model}/data/dofs_{approach}{test_case}.npy')
        qois = np.load(f'{model}/data/qois_{approach}{test_case}.npy')
    except IOError:
        print(f'Cannot load {approach} data for test case {test_case}')
        continue
    axes.semilogx(dofs, qois, '--x', label=label)
axes.set_xlabel('DoF count')
axes.set_ylabel('Quantity of Interest')
axes.grid(True)
axes.legend()
plt.tight_layout()
plt.savefig(f'{model}/plots/qoi_convergence{test_case}.pdf')
