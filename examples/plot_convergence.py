from plotting import *

import argparse
from matplotlib.ticker import FormatStrFormatter
import numpy as np


matplotlib.rcParams['font.size'] = 20

# Parse for test case
parser = argparse.ArgumentParser()
parser.add_argument('model', help='The model')
parsed_args = parser.parse_args()
model = parsed_args.model
assert model in ['turbine']
approaches = {
    'uniform': {'label': 'Uniform refinement', 'color': 'cornflowerblue', 'marker': 'x', 'linestyle': '-'},
    'GOisotropic': {'label': 'Isotropic goal-oriented adaptation', 'color': 'orange', 'marker': '^', 'linestyle': '-'},
    'GOanisotropic': {'label': 'Anisotropic goal-oriented adaptation', 'color': 'g', 'marker': 'o', 'linestyle': '-'},
    'MLisotropic': {'label': 'Isotropic data-driven adaptation', 'color': 'orange', 'marker': '^', 'linestyle': '--'},
    'MLanisotropic': {'label': 'Anisotropic data-driven adaptation', 'color': 'g', 'marker': 'o', 'linestyle': '--'},
}
xlim = [3.0e+03, 4.0e+06]

# Plot QoI curves
for test_case in range(16):
    fig, axes = plt.subplots()
    start = max(np.load(f'{model}/data/qois_uniform_{test_case}.npy'))
    conv = np.load(f'{model}/data/qois_GOanisotropic_{test_case}.npy')[-1]
    axes.hlines(conv, *xlim, 'k', label='Converged QoI')
    for approach, metadata in approaches.items():
        try:
            dofs = np.load(f'{model}/data/dofs_{approach}_{test_case}.npy')
            qois = np.load(f'{model}/data/qois_{approach}_{test_case}.npy')
        except IOError:
            print(f'Cannot load {approach} data for test case {test_case}')
            continue
        axes.semilogx(dofs, qois, **metadata)
    axes.set_xlim(xlim)
    axes.set_ylim([conv - 0.05*(start - conv), start + 0.05*(start - conv)])
    axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes.set_xlabel('DoF count')
    axes.set_ylabel(r'Power output ($\mathrm{MW}$)')
    axes.grid(True)
    plt.tight_layout()
    plt.savefig(f'{model}/plots/qoi_convergence{test_case}.pdf')
    if test_case < 15:
        plt.close()

# Plot legend
fig2, axes2 = plt.subplots()
lines, labels = axes.get_legend_handles_labels()
legend = axes2.legend(lines, labels, frameon=False, ncol=3)
fig2.canvas.draw()
axes2.set_axis_off()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
plt.savefig(f'{model}/plots/legend.pdf', bbox_inches=bbox)

# Plot QoI error curves
for test_case in range(16):
    fig, axes = plt.subplots()
    conv = np.load(f'{model}/data/qois_GOanisotropic_{test_case}.npy')[-1]
    for approach, metadata in approaches.items():
        try:
            dofs = np.load(f'{model}/data/dofs_{approach}_{test_case}.npy')
            errors = np.abs(np.load(f'{model}/data/qois_{approach}_{test_case}.npy') - conv)
        except IOError:
            print(f'Cannot load {approach} data for test case {test_case}')
            continue
        if approach == 'GOanisotropic':
            dofs = dofs[:-1]
            errors = errors[:-1]
        axes.loglog(dofs, errors, **metadata)
    axes.set_xlim(xlim)
    axes.set_xlabel('DoF count')
    axes.set_ylabel(r'QoI error ($\%$)')
    axes.grid(True)
    plt.tight_layout()
    plt.savefig(f'{model}/plots/qoi_error_convergence{test_case}.pdf')
    plt.close()
