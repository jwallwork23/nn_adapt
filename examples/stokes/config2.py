from setup import *
import os


def get_qoi(mesh_seq, i):
    def qoi(sol):
        x, y = SpatialCoordinate(mesh_seq[i])
        k = conditional(And(And(x > 14, x < 16), y > 6), 1, 0)
        u, p = sol['up'].split()
        return k*dot(u, u)*dx

    return qoi


def u_inflow(mesh):
    x, y = SpatialCoordinate(mesh)
    return as_vector([exp(1 - 1/(1 - ((y - 5)/5)**2)), 0])


parameters.viscosity = Constant(0.5)
parameters.u_inflow = u_inflow
qoi_name = "Speed above the circle"
