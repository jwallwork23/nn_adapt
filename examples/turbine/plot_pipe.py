from firedrake import *
from nn_adapt.plotting import *


matplotlib.rcParams["font.size"] = 12

# Plot mesh
fig, axes = plt.subplots(figsize=(6, 3))
mesh = Mesh("meshes/pipe.msh")
triplot(
    mesh,
    axes=axes,
    boundary_kw={"color": "dodgerblue"},
    interior_kw={"edgecolor": "w"},
)

# Add turbine footprints
P0 = FunctionSpace(mesh, "DG", 0)
footprints = assemble(sum(TestFunction(P0) * dx(tag) for tag in (2, 3)))
footprints.interpolate(conditional(footprints > 0, 0, 1))
tricontourf(footprints, axes=axes, cmap="Blues", levels=[0, 1])

# Annotate with physical parameters
txt = r"""$\nu = 100.0$
$b = 50.0$
$u_{\mathrm{in}} = \widetilde{y}^2(1-\widetilde{y})^2$
"""
xy = (1175, 10)
axes.annotate(txt, xy=xy, bbox={"fc": "w"})
axes.annotate(r"($\widetilde{y}=y/200$)", xy=xy, color="grey")

axes.axis(False)
plt.tight_layout()
plt.savefig("plots/pipe.pdf")
