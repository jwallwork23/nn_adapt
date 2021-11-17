import argparse
import matplotlib.pyplot as plt
import numpy as np


# Parse for test case
parser = argparse.ArgumentParser()
parser.add_argument('test_case', help='The configuration file number')
test_case = int(parser.parse_args().test_case)
assert test_case in [0, 1, 2, 3, 4]

# Load data
dofs = np.load(f'data/dofs_uniform{test_case}.npy')
qois = np.load(f'data/qois_uniform{test_case}.npy')

# Plot
fig, axes = plt.subplots()
axes.semilogx(dofs, qois, '--x')
axes.set_xlabel('DoF count')
axes.set_ylabel('Quantity of Interest')
axes.grid(True)
plt.tight_layout()
plt.savefig(f'plots/uniform_convergence{test_case}.pdf')
