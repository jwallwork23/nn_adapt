from firedrake import *
import matplotlib
import numpy as np


matplotlib.rcParams["font.size"] = 12


def plot_config(config, mesh, axes):
    """
    Plot a given configuration of a problem on a given
    mesh and axes.
    """
    tags = config.parameters.turbine_ids
    P0 = FunctionSpace(mesh, "DG", 0)
    footprints = assemble(sum(TestFunction(P0) * dx(tag) for tag in tags))
    footprints.interpolate(conditional(footprints > 0, 0, 1))
    triplot(
        mesh,
        axes=axes,
        boundary_kw={"color": "dodgerblue"},
        interior_kw={"edgecolor": "w"},
    )
    tricontourf(footprints, axes=axes, cmap="Blues", levels=[0, 1])

    # Bounding box
    xmin = 0
    xmax = 1200
    ymin = 0
    ymax = 500
    eps = 5

    # Adjust axes
    W = assemble(Constant(1.0, domain=mesh) * ds(1))
    L = 0.5 * assemble(
        Constant(1.0, domain=mesh) * ds(3)
    )  # NOTE: both top and bottom are tagged as 3
    dL = 0.5 * (xmax - L)
    dW = 0.5 * (ymax - W)
    axes.axis(False)
    axes.set_xlim([xmin - dL - eps, xmax - dL + eps])
    axes.set_ylim([ymin - dW - eps, ymax - dW + eps])

    # Annotate with viscosity coefficient and bathymetry
    nu = config.parameters.viscosity_coefficient
    b = config.parameters.depth
    u_in = config.parameters.inflow_speed
    txt = r"$\nu$ = %.3f, $b$ = %.2f, $u_{\mathrm{in}}$ = %.2f" % (nu, b, u_in)
    axes.annotate(
        txt, xy=(0.025 * L, -0.25 * W), bbox={"fc": "w"}, annotation_clip=False
    )


def process_sensitivities(data, layout):
    """
    Separate sensitivity experiment data by variable.

    :arg data: the output of `compute_importance.py`
    :arg layout: the :class:`NetLayout` instance
    """
    i = 0
    sensitivities = {}
    dofs = {"u": 3, "v": 3, r"\eta": 6}
    for label in ("physics", "mesh", "forward", "adjoint"):
        n = layout.count_inputs(label)
        if n == 0:
            continue
        if label in ("forward", "adjoint"):
            assert n == sum(dofs.values())
            for key, dof in dofs.items():
                S = np.zeros(6)
                for j in range(dof):
                    S[j] = data[i + j]
                l = (r"$%s$" if label == "forward" else r"$%s^*$") % key
                sensitivities[l] = S
                i += dof
        else:
            S = np.zeros(6)
            for j in range(n):
                S[j] = data[i + j]
            i += n
            sensitivities[label.capitalize()] = S
    return sensitivities
