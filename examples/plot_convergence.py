"""
Plot QoI convergence curves under uniform refinement,
goal-oriented mesh adaptation and data-driven mesh
adaptation, for a given ``test_case`` and ``model``.
"""
from nn_adapt.parse import Parser
from nn_adapt.plotting import *

import importlib
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def plot_slope(x0, x1, y0, g, axes):
    """
    Plot a slope marker for the gradient of a curve.
    """
    y1 = y0 * (x1 / x0) ** g
    axes.plot([x0, x1], [y0, y1], "-", color="darkgray")


matplotlib.rcParams["font.size"] = 20

# Parse user input
parser = Parser("plot_convergence.py")
parsed_args = parser.parse_args()
model = parsed_args.model
test_case = parsed_args.test_case
approaches = {
    "uniform": {
        "label": "Uniform refinement",
        "color": "cornflowerblue",
        "marker": "x",
        "linestyle": "-",
    },
    "GOisotropic": {
        "label": "Isotropic goal-oriented adaptation",
        "color": "orange",
        "marker": "^",
        "linestyle": "-",
    },
    "GOanisotropic": {
        "label": "Anisotropic goal-oriented adaptation",
        "color": "g",
        "marker": "o",
        "linestyle": "-",
    },
    "MLisotropic": {
        "label": "Isotropic data-driven adaptation",
        "color": "orange",
        "marker": "^",
        "linestyle": "--",
    },
    "MLanisotropic": {
        "label": "Anisotropic data-driven adaptation",
        "color": "g",
        "marker": "o",
        "linestyle": "--",
    },
}
xlim = [3.0e03, 4.0e06]

# Load configuration
setup = importlib.import_module(f"{model}.config")
unit = setup.parameters.qoi_unit
qoi_name = setup.parameters.qoi_name.capitalize()

dofs, elements, qois = {}, {}, {}
for approach in approaches:
    try:
        dofs[approach] = np.load(f"{model}/data/dofs_{approach}_{test_case}.npy")
        elements[approach] = np.load(f"{model}/data/elements_{approach}_{test_case}.npy")
        qois[approach] = np.load(f"{model}/data/qois_{approach}_{test_case}.npy")
    except IOError:
        print(f"Cannot load {approach} data for test case {test_case}")
        approaches.pop(approach)
        continue

# Plot QoI curves against DoF count
fig, axes = plt.subplots()
start = max(np.load(f"{model}/data/qois_uniform_{test_case}.npy"))
conv = np.load(f"{model}/data/qois_GOanisotropic_{test_case}.npy")[-1]
axes.hlines(conv, *xlim, "k", label="Converged QoI")
for approach, metadata in approaches.items():
    axes.semilogx(dofs[approach], qois[approach], **metadata)
axes.set_xlim(xlim)
axes.set_ylim([conv - 0.05 * (start - conv), start + 0.05 * (start - conv)])
axes.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
axes.set_xlabel("DoF count")
axes.set_ylabel(qoi_name + r" ($\mathrm{" + unit + r"}$)")
axes.grid(True)
plt.tight_layout()
plt.savefig(f"{model}/plots/qoi_vs_dofs_{test_case}.pdf")

# Plot legend
fig2, axes2 = plt.subplots()
lines, labels = axes.get_legend_handles_labels()
legend = axes2.legend(lines, labels, frameon=False, ncol=3)
fig2.canvas.draw()
axes2.set_axis_off()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
plt.savefig(f"{model}/plots/legend.pdf", bbox_inches=bbox)

# Plot QoI curves against element count
fig, axes = plt.subplots()
start = max(np.load(f"{model}/data/qois_uniform_{test_case}.npy"))
conv = np.load(f"{model}/data/qois_GOanisotropic_{test_case}.npy")[-1]
axes.hlines(conv, *xlim, "k", label="Converged QoI")
for approach, metadata in approaches.items():
    axes.semilogx(elements[approach], qois[approach], **metadata)
axes.set_ylim([conv - 0.05 * (start - conv), start + 0.05 * (start - conv)])
axes.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
axes.set_xlabel("Element count")
axes.set_ylabel(qoi_name + r" output ($\mathrm{" + unit + "}$)")
axes.grid(True)
plt.tight_layout()
plt.savefig(f"{model}/plots/qoi_vs_elements_{test_case}.pdf")
plt.close()

# Plot QoI error curves against DoF count
errors = {}
fig, axes = plt.subplots()
for approach, metadata in approaches.items():
    errors[approach] = np.abs(qois[approach] - conv)
    d = dofs[approach][:-1] if approach == "GOanisotropic" else dofs[approach]
    err = errors[approach][:-1] if approach == "GOanisotropic" else errors[approach]
    axes.loglog(d, err, **metadata)
axes.set_xlim(xlim)
axes.set_xlabel("DoF count")
axes.set_ylabel(r"QoI error ($\%$)")
axes.grid(True, which="both")
plt.tight_layout()
plt.savefig(f"{model}/plots/qoi_error_vs_dofs_{test_case}.pdf")
plt.close()

# Plot QoI error curves against element count
fig, axes = plt.subplots()
for approach, metadata in approaches.items():
    els = elements[approach][:-1] if approach == "GOanisotropic" else elements[approach]
    err = errors[approach][:-1] if approach == "GOanisotropic" else errors[approach]
    axes.loglog(els, err, **metadata)
axes.set_xlabel("Element count")
axes.set_ylabel(r"QoI error ($\%$)")
axes.grid(True, which="both")
plt.tight_layout()
plt.savefig(f"{model}/plots/qoi_error_vs_elements_{test_case}.pdf")
plt.close()
