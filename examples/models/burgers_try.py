from copy import deepcopy
from firedrake import *
from firedrake.petsc import PETSc
from firedrake_adjoint import *
from firedrake.adjoint import get_solve_blocks
import nn_adapt.model
import nn_adapt.solving

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

    # Timestepping parameters
    timestep = 0.05
    tt_steps = 10

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
        yside = self.initial_speed * sin(pi * y)
        yside = 0
        return as_vector([expr, yside])


PETSc.Sys.popErrorHandler()
parameters = Parameters()


def get_function_space(mesh):
    r"""
    Construct the :math:`\mathbb P2` finite element space
    used for the prognostic solution.
    """
    return VectorFunctionSpace(mesh, "CG", 2)


class Solver(nn_adapt.solving.Solver):
    """
    Solver object based on current mesh and state.
    """

    def __init__(self, mesh, ic, **kwargs):
        """
        :arg mesh: the mesh to define the solver on
        :arg ic: the current state / initial condition
        """
        self.mesh = mesh

        # Collect parameters
        dt = Constant(parameters.timestep)
        nu = parameters.viscosity(mesh)

        # Define variational formulation
        V = self.function_space
        u = Function(V)
        u_ = Function(V)
        v = TestFunction(V)
        self._form = (
            inner((u - u_) / dt, v) * dx
            + inner(dot(u, nabla_grad(u)), v) * dx
            + nu * inner(grad(u), grad(v)) * dx
        )
        problem = NonlinearVariationalProblem(self._form, u)

        # Set initial condition
        u_.project(parameters.ic(mesh))

        # Create solver
        self._solver = NonlinearVariationalSolver(problem)
        self._solution = u

    @property
    def function_space(self):
        r"""
        The :math:`\mathbb P2` finite element space.
        """
        return get_function_space(self.mesh)

    @property
    def form(self):
        """
        The weak form of Burgers equation
        """
        return self._form

    @property
    def solution(self):
        return self._solution

    def iterate(self, **kwargs):
        """
        Take a single timestep of Burgers equation
        """
        self._solver.solve()
        
        
class time_dependent_Solver(nn_adapt.solving.Solver):
    """
    Solver object based on current mesh and state.
    """

    def __init__(self, meshes, ic, **kwargs):
        """
        :arg mesh: the mesh to define the solver on
        :arg ic: the current state / initial condition
        """
        self.meshes = meshes

        # Collect parameters
        self.tt_steps = parameters.tt_steps
        self.dt = Constant(parameters.timestep)
        assert self.tt_steps == len(self.meshes)
        
        # Physical parameters
        self.nu = Constant(parameters.viscosity_coefficient)

    @property
    def function_space(self):
        r"""
        The :math:`\mathbb P2` finite element space.
        """
        return get_function_space(self.meshes)

    @property
    def form(self):
        """
        The weak form of Burgers equation
        """
        return self._form

    @property
    def solution(self):
        return self._solutions
    
    def adjoint_setup(self):
        J_form = inner(self._u, self._u)*ds(2)
        J = assemble(J_form)

        tape = get_working_tape()
        g = compute_gradient(J, Control(self.nu))

        solve_blocks = get_solve_blocks()
            
        # 'Initial condition' for both adjoint
        dJdu = assemble(derivative(J_form, self._u))
        
        return dJdu, solve_blocks

    def iterate(self, **kwargs):
        """
        Get the final timestep of Burgers equation
        """
        stop_annotating();

        # Assign initial condition
        V = get_function_space(self.meshes[0])
        ic = parameters.ic(self.meshes[0])
        u = Function(V)
        u.project(ic)

        _solutions = [u.copy(deepcopy=True)]

        # solve forward
        step = 0

        for step in range(self.tt_steps):
            # Define P2 function space and corresponding test function
            V = get_function_space(self.meshes[step])
            v = TestFunction(V)

            # Create Functions for the solution and time-lagged solution
            u_ = Function(V)
            u_.project(u)
            
            u = Function(V, name="Velocity")

            # Define nonlinear form
            F = (inner((u - u_)/self.dt, v) + inner(dot(u, nabla_grad(u)), v) + self.nu*inner(grad(u), grad(v)))*dx

            solve(F == 0, u)

            # Store forward solution at exports so we can plot again later
            _solutions.append(u.copy(deepcopy=True))
        
        self._form = F
        self._solutions = _solutions
        self._u = u
        
        stop_annotating();
        


def get_initial_condition(function_space):
    """
    Compute an initial condition based on the initial
    speed parameter.
    """
    u = Function(function_space)
    u.interpolate(parameters.ic(function_space.mesh()))
    return u


def get_qoi(mesh):
    """
    Extract the quantity of interest function from the :class:`Parameters`
    object.

    It should have one argument - the prognostic solution.
    """
    
    def qoi(sol):
        sol_temp = Function(mesh)
        sol_temp.project(sol)
        return inner(sol_temp, sol_temp) * ds(2)

    return qoi


# # Initial mesh for all test cases
# initial_mesh = [UnitSquareMesh(30, 30), UnitSquareMesh(50, 30)]


# # A simple pretest
# a = time_dependent_Solver(meshes = initial_mesh, ic = 0, kwargs='0')
# a.iterate()
# b = a.solution
# import matplotlib.pyplot as plt

# # fig, axes = plt.subplots(20)
# # for i in range(20):
# #     tricontourf(b[i], axes=axes[i])

# fig, axes = plt.subplots(2)
# tricontourf(b[0], axes=axes[0])
# tricontourf(b[1], axes=axes[1])

# plt.show()
