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
import os
import sys


# Parse user input
parser = Parser("plot_convergence.py")
parsed_args = parser.parse_args()
model = parsed_args.model
test_case = parsed_args.test_case

# Formatting
matplotlib.rcParams["font.size"] = 20
approaches = {
    "uniform": {
        "label": "Uniform refinement",
        "color": "cornflowerblue",
        "marker": "x",
        "linestyle": "-",
    },
    "GOanisotropic": {
        "label": "Goal-oriented adaptation",
        "color": "orange",
        "marker": "o",
        "linestyle": "-",
    },
    "MLanisotropic": {
        "label": "Data-driven adaptation",
        "color": "g",
        "marker": "^",
        "linestyle": "-",
    },
}
xlim = {
    "dofs": [3.0e03, 4.0e06],
    "elements": [3.0e02, 4.0e05],
    "times": [1.0e0, 2.0e03],
}

# Load configuration
setup = importlib.import_module(f"{model}.config")
unit = setup.parameters.qoi_unit
qoi_name = setup.parameters.qoi_name.capitalize()

# Load outputs
dofs, elements, qois, times, niter = {}, {}, {}, {}, {}
for approach in approaches.copy():
    try:
        dofs[approach] = np.load(f"{model}/data/dofs_{approach}_{test_case}.npy")
        elements[approach] = np.load(
            f"{model}/data/elements_{approach}_{test_case}.npy"
        )
        qois[approach] = np.load(f"{model}/data/qois_{approach}_{test_case}.npy")
        times[approach] = np.load(f"{model}/data/times_{approach}_{test_case}.npy")
        niter[approach] = np.load(f"{model}/data/niter_{approach}_{test_case}.npy")
        print(f"Iteration count for {approach}: {niter[approach]}")
    except IOError:
        print(f"Cannot load {approach} data for test case {test_case}")
        approaches.pop(approach)
        continue
if len(approaches.keys()) == 0:
    print("Nothing to plot.")
    sys.exit(0)

# Plot QoI curves against DoF count
fig, axes = plt.subplots()
start = max(np.load(f"{model}/data/qois_uniform_{test_case}.npy"))
conv = 19.82480044 if test_case == "aligned" else 23.17004884  # TODO: avoid hard-code
# conv = np.load(f"{model}/data/qois_GOanisotropic_{test_case}.npy")[-1]
axes.hlines(conv, *xlim["dofs"], "k", label="Converged QoI")
for approach, metadata in approaches.items():
    axes.semilogx(dofs[approach], qois[approach], **metadata)
axes.set_xlim(xlim["dofs"])
axes.set_ylim([conv - 0.05 * (start - conv), start + 0.05 * (start - conv)])
axes.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
axes.set_xlabel("DoF count")
axes.set_ylabel(qoi_name + r" ($\mathrm{" + unit + r"}$)")
axes.grid(True)
plt.tight_layout()
plt.savefig(f"{model}/plots/qoi_vs_dofs_{test_case}.pdf")

# Plot QoI curves against CPU time
fig, axes = plt.subplots()
axes.hlines(conv, *xlim["times"], "k", label="Converged QoI")
for approach, metadata in approaches.items():
    axes.semilogx(times[approach], qois[approach], **metadata)
    for n, t, q in zip(niter[approach], times[approach], qois[approach]):
        axes.annotate(str(n), (1.1 * t, q), color=metadata["color"], fontsize=14)
axes.set_xlim(xlim["times"])
axes.set_ylim([conv - 0.05 * (start - conv), start + 0.05 * (start - conv)])
axes.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
axes.set_xlabel(r"CPU time ($\mathrm{s}$)")
axes.set_ylabel(qoi_name + r" ($\mathrm{" + unit + "}$)")
axes.grid(True)
plt.tight_layout()
plt.savefig(f"{model}/plots/qoi_vs_cputime_{test_case}.pdf")
plt.close()

# Plot QoI error curves against DoF count
errors = {}
fig, axes = plt.subplots()
for approach, metadata in approaches.items():
    errors[approach] = np.abs((qois[approach] - conv) / conv)
    axes.loglog(dofs[approach], errors[approach], **metadata)
axes.set_xlim(xlim["dofs"])
axes.set_xlabel("DoF count")
axes.set_ylabel(r"QoI error ($\%$)")
axes.grid(True, which="both")
plt.tight_layout()
plt.savefig(f"{model}/plots/qoi_error_vs_dofs_{test_case}.pdf")
plt.close()

# Plot legend
fname = f"{model}/plots/legend.pdf"
if not os.path.exists(fname):
    fig2, axes2 = plt.subplots()
    lines, labels = axes.get_legend_handles_labels()
    legend = axes2.legend(lines, labels, frameon=False, ncol=3)
    fig2.canvas.draw()
    axes2.set_axis_off()
    bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
    plt.savefig(fname, bbox_inches=bbox)

# Plot QoI error curves against CPU time
fig, axes = plt.subplots()
for approach, metadata in approaches.items():
    axes.loglog(times[approach], errors[approach], **metadata)
    for n, t, e in zip(niter[approach], times[approach], errors[approach]):
        axes.annotate(str(n), (1.1 * t, e), color=metadata["color"], fontsize=14)
axes.set_xlim(xlim["times"])
axes.set_xlabel(r"CPU time ($\mathrm{s}$)")
axes.set_ylabel(r"QoI error ($\%$)")
axes.grid(True, which="both")
plt.tight_layout()
plt.savefig(f"{model}/plots/qoi_error_vs_cputime_{test_case}.pdf")
plt.close()

# Plot CPU time curves against DoF count
fig, axes = plt.subplots()
for approach, metadata in approaches.items():
    axes.loglog(dofs[approach], times[approach], **metadata)
    for n, t, d in zip(niter[approach], times[approach], dofs[approach]):
        axes.annotate(str(n), (1.1 * d, t), color=metadata["color"], fontsize=14)
axes.set_xlabel("DoF count")
axes.set_ylabel(r"CPU time ($\mathrm{s}$)")
axes.set_xlim(xlim["dofs"])
axes.set_ylim(xlim["times"])
axes.grid(True, which="both")
plt.tight_layout()
plt.savefig(f"{model}/plots/cputime_vs_dofs_{test_case}.pdf")
plt.close()
