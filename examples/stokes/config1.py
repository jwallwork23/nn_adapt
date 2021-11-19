from models.stokes import *


def get_qoi(mesh_seq, i):
    def qoi(sol):
        u, p = sol['up'].split()
        return p*ds(4)

    return qoi


def u_inflow(mesh):
    x, y = SpatialCoordinate(mesh)
    return as_vector([y*(10 - y)/25, 0])


parameters.viscosity = Constant(1.0)
parameters.u_inflow = u_inflow
qoi_name = "Pressure on the circle"
