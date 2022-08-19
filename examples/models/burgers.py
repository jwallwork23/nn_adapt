from firedrake import *
from firedrake.petsc import PETSc
from pyroteus import *
import pyroteus.go_mesh_seq
import nn_adapt.model
import nn_adapt.solving


class Parameters(nn_adapt.model.Parameters):
    """
    Class encapsulating all parameters required for a simple
    Burgers equation test case.
    """

    qoi_name = "right boundary integral"
    qoi_unit = r"m\,s^{-1}"

    # Adaptation parameters
    h_min = 1.0e-10  # Minimum metric magnitude
    h_max = 1.0  # Maximum metric magnitude

    # Physical parameters
    viscosity_coefficient = 0.0001
    initial_speed = 1.0

    # Timestepping parameters
    timestep = 0.05

    solver_parameters = {}
    adjoint_solver_parameters = {}

    def bathymetry(self, mesh):
        """
        Compute the bathymetry field on the current `mesh`.

        Note that there isn't really a concept of bathymetry
        for Burgers equation. It is kept constant and should
        be ignored by the network.
        """
        P0_2d = FunctionSpace(mesh, "DG", 0)
        return Function(P0_2d).assign(1.0)

    def drag(self, mesh):
        """
        Compute the bathymetry field on the current `mesh`.

        Note that there isn't really a concept of bathymetry
        for Burgers equation. It is kept constant and should
        be ignored by the network.
        """
        P0_2d = FunctionSpace(mesh, "DG", 0)
        return Function(P0_2d).assign(1.0)

    def viscosity(self, mesh):
        """
        Compute the viscosity coefficient on the current `mesh`.
        """
        P0_2d = FunctionSpace(mesh, "DG", 0)
        return Function(P0_2d).assign(self.viscosity_coefficient)

    def ic(self, mesh):
        """
        Initial condition
        """
        x, y = SpatialCoordinate(mesh)
        expr = self.initial_speed * sin(pi * x)
        return as_vector([expr, 0])


PETSc.Sys.popErrorHandler()
parameters = Parameters()


def get_function_spaces(mesh):
    r"""
    Construct the :math:`\mathbb P2` finite element spaces
    used for the prognostic solution.
    """
    return {"u": VectorFunctionSpace(mesh, "CG", 2)}


def get_form(mesh_seq):
    def form(i, solutions):
        u, u_ = solutions["u"]
        P = mesh_seq.time_partition
        dt = Constant(P.timesteps[i])
        mesh = mesh_seq[i]
        nu = parameters.viscosity(mesh)

        # Setup variational problem
        v = TestFunction(u.function_space())
        F = (
            inner((u - u_) / dt, v) * dx
            + inner(dot(u, nabla_grad(u)), v) * dx
            + nu * inner(grad(u), grad(v)) * dx
        )
        return F

    return form


def get_solver(mesh_seq):
    def solver(i, ic):
        V = mesh_seq.function_spaces["u"][i]
        u = Function(V)

        # Set initial condition
        u_ = Function(V, name="u_old")
        u_.assign(ic["u"])

        # Define form
        F = mesh_seq.form(i, {"u": (u, u_)})

        # Solve
        solve(F == 0, u, ad_block_tag="u")
        return {"u": u}

    return solver


def get_initial_condition(mesh_seq):
    """
    Compute an initial condition based on the initial
    speed parameter.
    """
    function_space = mesh_seq.function_spaces["u"][0]
    u = Function(function_space)
    u.project(parameters.ic(function_space.mesh()))
    return {"u": u}


def get_qoi(mesh_seq, solutions, i):
    def end_time_qoi():
        u = solutions["u"]
        return inner(u, u) * ds(2)

    return end_time_qoi


# Initial mesh for all test cases
initial_mesh = UnitSquareMesh(30, 30)


def GoalOrientedMeshSeq(mesh, **kwargs):
    dt = parameters.timestep
    time_partition = TimeInterval(dt, dt, ["u"])
    mesh_seq = pyroteus.go_mesh_seq.GoalOrientedMeshSeq(
        time_partition,
        mesh,
        get_function_spaces=get_function_spaces,
        get_initial_condition=get_initial_condition,
        get_form=get_form,
        get_solver=get_solver,
        get_qoi=get_qoi,
        qoi_type="end_time",
    )
    return mesh_seq
