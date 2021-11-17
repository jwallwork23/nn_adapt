from setup import *
import os


def get_qoi(mesh_seq, i):
    def qoi(sol):
        x, y = SpatialCoordinate(mesh_seq[i])
        k = conditional(y < 5, 1, 0)
        u, p = sol['up'].split()
        return k*p*ds(4)

    return qoi


def u_inflow(mesh):
    x, y = SpatialCoordinate(mesh)
    return as_vector([27*y**2*(10 - y)/4000, 0])


parameters.viscosity = Constant(2.0)
parameters.u_inflow = u_inflow
mesh = Mesh(os.path.join(os.path.abspath(os.path.dirname(__file__)), "meshes/4.msh"))
qoi_name = "Pressure on the lower semicircle"
