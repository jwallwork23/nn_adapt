from firedrake import *
from pyroteus import *
import pyroteus.go_mesh_seq
from firedrake.petsc import PETSc
import nn_adapt.model

'''
A memory hungry method solving time dependent PDE.
'''
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
    
    # Offset for creating more initial conditions
    x_offset = 0
    y_offset = 0

    # Timestepping parameters
    timestep = 5

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
        # x_expr = self.initial_speed * sin(pi * x + self.x_offset)
        # y_expr = self.initial_speed * sin(pi * y + self.y_offset)
        x_expr = self.initial_speed * sin(pi * x)
        y_expr = 0
        return as_vector([x_expr, y_expr])
    
    
def get_function_spaces(mesh):
    return {"u": VectorFunctionSpace(mesh, "CG", 2)}


def get_form(mesh_seq):
    def form(index, solutions):
        u, u_ = solutions["u"]
        P = mesh_seq.time_partition
        dt = Constant(P.timesteps[index])

        # Specify viscosity coefficient
        nu = parameters.viscosity_coefficient

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
    def solver(index, ic):
        function_space = mesh_seq.function_spaces["u"][index]
        u = Function(function_space)

        # Initialise 'lagged' solution
        u_ = Function(function_space, name="u_old")
        u_.assign(ic["u"])

        # Define form
        F = mesh_seq.form(index, {"u": (u, u_)})

        # Time integrate from t_start to t_end
        P = mesh_seq.time_partition
        t_start, t_end = P.subintervals[index]
        dt = P.timesteps[index]
        t = t_start
        step = 0
        sp = {'snes_max_it': 100}
        while t < t_end - 1.0e-05:
            step += 1
            solve(F == 0, u, ad_block_tag="u", solver_parameters=sp)
            u_.assign(u)
            t += dt
        return {"u": u}

    return solver


def get_initial_condition(mesh_seq):
    fs = mesh_seq.function_spaces["u"][0]
    return {"u": interpolate(parameters.ic(fs), fs)}

def get_qoi(mesh_seq, solutions, index):
    def end_time_qoi():
        u = solutions["u"]
        fs = mesh_seq.function_spaces["u"][0]
        x, y = SpatialCoordinate(fs)
        partial = conditional(And(And(x > 0.1, x < 0.9), And(y > 0.45, y < 0.55)), 1, 0)
        return inner(u, u) * dx

    def time_integrated_qoi(t):
        fs = mesh_seq.function_spaces["u"][0]
        x, y = SpatialCoordinate(fs)
        partial = conditional(And(And(x > 0.7, x < 0.8), And(y > 0.45, y < 0.55)), 1, 0)
        dt = Constant(mesh_seq.time_partition[index].timestep)
        u = solutions["u"]
        return dt * partial * inner(u, u) * dx
    
    if mesh_seq.qoi_type == "end_time":
        return end_time_qoi
    else:
        return time_integrated_qoi


PETSc.Sys.popErrorHandler()
parameters = Parameters()

def GoalOrientedMeshSeq(mesh, **kwargs):
    fields = ["u"]
    
    try:
        num_subintervals = len(mesh)
    except:
        num_subintervals = 1

    # setup time steps and export steps
    dt = 1
    steps_subintervals = 10
    end_time = num_subintervals * steps_subintervals * dt
    timesteps_per_export = 1
    
    # setup pyroteus time_partition
    time_partition = TimePartition(
        end_time,
        num_subintervals,
        dt,
        fields,
        timesteps_per_export=timesteps_per_export,
        )
    
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
    

initial_mesh = lambda: UnitSquareMesh(30, 30)
        