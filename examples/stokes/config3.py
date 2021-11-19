from models.stokes import *


def get_qoi(mesh_seq, i):
    def qoi(sol):
        x, y = SpatialCoordinate(mesh_seq[i])
        u, p = sol['up'].split()
        k = conditional(x > 20, 1, 0)
        return k*inner(u, u)*1.5*dx

    return qoi


def u_inflow(mesh):
    x, y = SpatialCoordinate(mesh)
    return as_vector([y*(10 - y)/10.0, 0])


parameters.viscosity = Constant(0.25)
parameters.u_inflow = u_inflow
qoi_name = "Cubed speed downstream"
