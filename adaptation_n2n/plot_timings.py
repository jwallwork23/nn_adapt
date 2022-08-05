from nn_adapt.parse import Parser, nonnegative_int
from nn_adapt.plotting import *
import numpy as np


def get_times(model, approach, case, it, tag=None):
    """
    Gather the timing data for some approach applied
    to a given test case.

    :arg model: the PDE being solved
    :arg approach: the mesh adaptation approach
    :arg case: the test case name or number
    :arg it: the run
    :kwarg tag: the tag for the network
    """
    ext = f"_{tag}" if approach[:2] == "ML" else ""
    qoi = np.load(f"{model}/data/qois_{approach}_{case}{ext}.npy")[it]
    conv = np.load(f"{model}/data/qois_uniform_{case}.npy")[-1]
    print(f"{approach} QoI error: {abs((qoi-conv)/conv)*100:.3f} %")
    split = {
        "Forward solve": np.load(f"{model}/data/times_forward_{approach}_{case}{ext}.npy")[it],
        "Adjoint solve": np.load(f"{model}/data/times_adjoint_{approach}_{case}{ext}.npy")[it],
        "Error estimation": np.load(f"{model}/data/times_estimator_{approach}_{case}{ext}.npy")[it],
        "Metric construction": np.load(f"{model}/data/times_metric_{approach}_{case}{ext}.npy")[it],
        "Mesh adaptation": np.load(f"{model}/data/times_adapt_{approach}_{case}{ext}.npy")[it],
    }
    total = sum(split.values())
    for key, value in split.items():
        print(f"{approach} {key}: {value/total*100:.3f} %")
    niter = np.load(f"{model}/data/niter_{approach}_{case}{ext}.npy")[it]
    print(f"niter = {niter}")
    return split


# Parse user input
parser = Parser(prog="plot_timings.py")
parser.parse_tag()
parser.add_argument(
    "--iter",
    help="Iteration",
    type=nonnegative_int,
    default=21,
)
parsed_args, unknown_args = parser.parse_known_args()
model = parsed_args.model
try:
    test_case = int(parsed_args.test_case)
    assert test_case > 0
except ValueError:
    test_case = parsed_args.test_case
tag = parsed_args.tag
it = parsed_args.iter
approaches = ["GOanisotropic", "MLanisotropic"]

# Plot bar chart
fig, axes = plt.subplots(figsize=(6, 4.5))
colours = ["C0", "deepskyblue", "mediumturquoise", "mediumseagreen", "darkgreen", "0.3"]
data = {
    "Goal-oriented": get_times(model, "GOanisotropic", test_case, it, tag=tag),
    "Data-driven": get_times(model, "MLanisotropic", test_case, it, tag=tag),
}
bottom = np.zeros(len(data.keys()))
for i, key in enumerate(data["Goal-oriented"].keys()):
    arr = np.array([d[key] for d in data.values()])
    axes.bar(data.keys(), arr, bottom=bottom, label=key, color=colours[i])
    bottom += arr
axes.bar_label(axes.containers[-1])
axes.legend(loc="upper right")
axes.set_ylabel("Runtime [seconds]")
plt.tight_layout()
plt.savefig(f"{model}/plots/timings_{test_case}_{it}_{tag}.pdf")
