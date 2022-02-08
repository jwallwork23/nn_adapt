"""
Plot QoI convergence curves under uniform refinement,
goal-oriented mesh adaptation and data-driven mesh
adaptation, for a given ``test_case`` and ``model``.
"""
from nn_adapt.plotting import *

import argparse
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

# Parse for test case
parser = argparse.ArgumentParser(prog="plot_convergence.py")
parser.add_argument("model", help="The model")
parser.add_argument("test_case", help="The configuration name")
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

# Plot QoI curves against DoF count
fig, axes = plt.subplots()
start = max(np.load(f"{model}/data/qois_uniform_{test_case}.npy"))
conv = np.load(f"{model}/data/qois_GOanisotropic_{test_case}.npy")[-1]
axes.hlines(conv, *xlim, "k", label="Converged QoI")
for approach, metadata in approaches.items():
    try:
        dofs = np.load(f"{model}/data/dofs_{approach}_{test_case}.npy")
        qois = np.load(f"{model}/data/qois_{approach}_{test_case}.npy")
    except IOError:
        print(f"Cannot load {approach} data for test case {test_case}")
        continue
    axes.semilogx(dofs, qois, **metadata)
axes.set_xlim(xlim)
axes.set_ylim([conv - 0.05 * (start - conv), start + 0.05 * (start - conv)])
axes.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
axes.set_xlabel("DoF count")
axes.set_ylabel(r"{" + qoi_name + "} ($\mathrm{" + unit + "}$)")
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
    try:
        elements = np.load(f"{model}/data/elements_{approach}_{test_case}.npy")
        qois = np.load(f"{model}/data/qois_{approach}_{test_case}.npy")
    except IOError:
        print(f"Cannot load {approach} data for test case {test_case}")
        continue
    axes.semilogx(elements, qois, **metadata)
axes.set_ylim([conv - 0.05 * (start - conv), start + 0.05 * (start - conv)])
axes.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
axes.set_xlabel("Element count")
axes.set_ylabel(qoi_name + r" output ($\mathrm{" + unit + "}$)")
axes.grid(True)
plt.tight_layout()
plt.savefig(f"{model}/plots/qoi_vs_elements_{test_case}.pdf")
plt.close()

# Plot QoI error curves against DoF count
fig, axes = plt.subplots()
conv = np.load(f"{model}/data/qois_GOanisotropic_{test_case}.npy")[-1]
for approach, metadata in approaches.items():
    try:
        dofs = np.load(f"{model}/data/dofs_{approach}_{test_case}.npy")
        errors = np.abs(np.load(f"{model}/data/qois_{approach}_{test_case}.npy") - conv)
    except IOError:
        print(f"Cannot load {approach} data for test case {test_case}")
        continue
    if approach == "GOanisotropic":
        dofs = dofs[:-1]
        errors = errors[:-1]
    axes.loglog(dofs, errors, **metadata)
axes.set_xlim(xlim)
axes.set_xlabel("DoF count")
axes.set_ylabel(r"QoI error ($\%$)")
axes.grid(True, which="both")
plt.tight_layout()
plt.savefig(f"{model}/plots/qoi_error_vs_dofs_{test_case}.pdf")
plt.close()

# Plot QoI error curves against element count
fig, axes = plt.subplots()
conv = np.load(f"{model}/data/qois_GOanisotropic_{test_case}.npy")[-1]
for approach, metadata in approaches.items():
    try:
        elements = np.load(f"{model}/data/elements_{approach}_{test_case}.npy")
        errors = np.abs(np.load(f"{model}/data/qois_{approach}_{test_case}.npy") - conv)
    except IOError:
        print(f"Cannot load {approach} data for test case {test_case}")
        continue
    if approach == "GOanisotropic":
        elements = elements[:-1]
        errors = errors[:-1]
    axes.loglog(elements, errors, **metadata)
axes.set_xlabel("Element count")
axes.set_ylabel(r"QoI error ($\%$)")
axes.grid(True, which="both")
plt.tight_layout()
plt.savefig(f"{model}/plots/qoi_error_vs_elements_{test_case}.pdf")
plt.close()
