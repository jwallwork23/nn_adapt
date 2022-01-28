from firedrake import *
import matplotlib


matplotlib.rcParams['font.size'] = 12


def plot_config(config, mesh, axes):
    """
    Plot a given configuration of a problem on a given
    mesh and axes.
    """
    tags = config.parameters.turbine_ids
    P0 = FunctionSpace(mesh, 'DG', 0)
    footprints = assemble(sum(TestFunction(P0)*dx(tag) for tag in tags))
    footprints.interpolate(conditional(footprints > 0, 0, 1))
    triplot(mesh, axes=axes, boundary_kw={'color': 'dodgerblue'}, interior_kw={'edgecolor': 'w'})
    tricontourf(footprints, axes=axes, cmap='Blues', levels=[0, 1])

    # Bounding box
    xmin = 0
    xmax = 1200
    ymin = 0
    ymax = 500
    eps = 5

    # Adjust axes
    W = assemble(Constant(1.0, domain=mesh)*ds(1))
    L = 0.5*assemble(Constant(1.0, domain=mesh)*ds(3))  # NOTE: both top and bottom are tagged as 3
    dL = 0.5*(xmax-L)
    dW = 0.5*(ymax-W)
    axes.axis(False)
    axes.set_xlim([xmin - dL - eps, xmax - dL + eps])
    axes.set_ylim([ymin - dW - eps, ymax - dW + eps])

    # Annotate with viscosity coefficient and bathymetry
    nu = config.parameters.viscosity.values()[0]
    b = config.parameters.depth
    u_in = config.parameters.inflow_speed
    axes.annotate(r'$\nu$' + f' = {nu:.3f}', xy=(0.7*L, 0.85*W), color='darkgrey')
    axes.annotate(r'$b$' + f' = {b:.2f}', xy=(0.7*L, 0.7*W), color='darkgrey')
    axes.annotate(r'$u_{\mathrm{in}}$' + f' = {u_in:.2f}', xy=(0.7*L, 0.55*W), color='darkgrey')
