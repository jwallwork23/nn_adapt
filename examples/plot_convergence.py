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
    "dofs": [3.0e03, 3.0e06],
    "times": [1.0e0, 2.0e03],
}

# Load configuration
setup = importlib.import_module(f"{model}.config")
setup.initialise(test_case)
unit = setup.parameters.qoi_unit
qoi_name = setup.parameters.qoi_name.capitalize()

# Load outputs
dofs, qois, times, niter = {}, {}, {}, {}
for approach in approaches.copy():
    try:
        dofs[approach] = np.load(f"{model}/data/dofs_{approach}_{test_case}.npy")
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

# Drop first iteration because timings include compilation   # FIXME: Why?
dofs["uniform"] = dofs["uniform"][1:]
qois["uniform"] = qois["uniform"][1:]
times["uniform"] = times["uniform"][1:]
niter["uniform"] = niter["uniform"][1:]

# Plot QoI curves against DoF count
fig, axes = plt.subplots()
start = max(np.load(f"{model}/data/qois_uniform_{test_case}.npy"))
conv = np.load(f"{model}/data/qois_uniform_{test_case}.npy")[-1]
axes.hlines(conv, *xlim["dofs"], "k", label="Converged QoI")
for approach, metadata in approaches.items():
    axes.semilogx(dofs[approach], qois[approach], **metadata)
axes.set_xlim(xlim["dofs"])
if test_case in ["aligned", "offset"]:
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
if test_case in ["aligned", "offset"]:
    axes.set_ylim([conv - 0.05 * (start - conv), start + 0.05 * (start - conv)])
axes.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
axes.set_xlabel(r"CPU time ($\mathrm{s}$)")
axes.set_ylabel(qoi_name + r" ($\mathrm{" + unit + "}$)")
axes.grid(True)
plt.tight_layout()
plt.savefig(f"{model}/plots/qoi_vs_cputime_{test_case}.pdf")
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

qois["uniform"] = qois["uniform"][:-1]
dofs["uniform"] = dofs["uniform"][:-1]
times["uniform"] = times["uniform"][:-1]

# Plot QoI error curves against DoF count
errors = {}
fig, axes = plt.subplots()
for approach, metadata in approaches.items():
    errors[approach] = np.abs((qois[approach] - conv) / conv)
    x, y = dofs[approach], errors[approach]
    a, b = np.polyfit(np.log(x), np.log(y), 1)
    print(f"QoI error vs. DoFs {approach}: gradient {a:.2f}")
    axes.scatter(x, y, **metadata)
    axes.loglog(x, x ** a * np.exp(b), color=metadata["color"])
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
    x, y = times[approach], errors[approach]
    if approach == "uniform":
        a, b = np.polyfit(np.log(x), np.log(y), 1)
        print(f"QoI error vs. time {approach}: gradient {a:.2f}")
        axes.loglog(x, x ** a * np.exp(b), color=metadata["color"])
    axes.scatter(x, y, **metadata)
    for n, t, e in zip(niter[approach], x, errors[approach]):
        axes.annotate(str(n), (1.1 * t, e), color=metadata["color"], fontsize=14)
axes.set_xlabel(r"CPU time ($\mathrm{s}$)")
axes.set_ylabel(r"QoI error ($\%$)")
axes.grid(True, which="both")
plt.tight_layout()
plt.savefig(f"{model}/plots/qoi_error_vs_cputime_{test_case}.pdf")
plt.close()
