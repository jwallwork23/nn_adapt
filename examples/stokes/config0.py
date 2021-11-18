from setup import *


def get_qoi(mesh_seq, i):
    def qoi(sol):
        u, p = sol['up'].split()
        return dot(u, u)*dx

    return qoi


def u_inflow(mesh):
    x, y = SpatialCoordinate(mesh)
    return as_vector([exp(-((x - 5)**2 + (y - 5)**2)/25), 0])


parameters.viscosity = Constant(0.1)
parameters.u_inflow = u_inflow
qoi_name = "Speed over the domain"
