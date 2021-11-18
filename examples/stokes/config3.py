from setup import *
import os


def get_qoi(mesh_seq, i):
    def qoi(sol):
        x, y = SpatialCoordinate(mesh_seq[i])
        u, p = sol['up'].split()
        k = conditional(And(And(x > 19, x < 21), And(y > 0.5, y < 2.5)), 1, 0)
        return k*inner(u, u)*dx

    return qoi


def u_inflow(mesh):
    x, y = SpatialCoordinate(mesh)
    return as_vector([y*(15 - y)/10.0, 0])


parameters.viscosity = Constant(0.25)
parameters.u_inflow = u_inflow
qoi_name = "Cubed speed below the circle"
