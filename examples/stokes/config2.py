from setup import *


def get_qoi(mesh_seq, i):
    def qoi(sol):
        x, y = SpatialCoordinate(mesh_seq[i])
        k = conditional(x < 15, 1, 0)
        u, p = sol['up'].split()
        return k*p**2*dx

    return qoi


def u_inflow(mesh):
    x, y = SpatialCoordinate(mesh)
    return as_vector([exp(1 - 1/(1 - ((y - 5)/5)**2)), 0])


parameters.viscosity = Constant(0.5)
parameters.u_inflow = u_inflow
qoi_name = "Upstream squared pressure"
