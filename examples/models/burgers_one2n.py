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
        
        
class Solver_one2n(nn_adapt.solving.Solver):
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
        self.tt_steps = parameters.tt_steps
        dt = Constant(parameters.timestep)
        
        # Physical parameters
        nu = parameters.viscosity(mesh)
        self.nu = nu
        
        # Define variational formulation
        V = self.function_space
        self.u = Function(V)
        self.u_ = Function(V)
        self.v = TestFunction(V)
        
        self._form = (
            inner((self.u - self.u_) / dt, self.v) * dx
            + inner(dot(self.u, nabla_grad(self.u)), self.v) * dx
            + nu * inner(grad(self.u), grad(self.v)) * dx
        )
        
        # Define initial conditions
        ic = parameters.ic(self.mesh)
        self.u.project(ic)
        
        # Set solutions
        self._solutions = []

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
        return self._solutions
    
    @property
    def adj_solution(self):
        return self._adj_solution
    
    def adjoint_iteration(self):
        """
        Get the forward solutions of Burgers equation
        """
        J_form = inner(self.u, self.u)*ds(2)
        J = assemble(J_form)

        g = compute_gradient(J, Control(self.nu))

        solve_blocks = get_solve_blocks()
            
        # 'Initial condition' for both adjoint
        dJdu = assemble(derivative(J_form, self.u))
        
        self._adj_solution = []
        for step in range(self.tt_steps-1):
            adj_sol = solve_blocks[step].adj_sol
            self._adj_solution.append(adj_sol)
        self._adj_solution.append(dJdu)

    def iterate(self, **kwargs):
        """
        Get the forward solutions of Burgers equation
        """
        tape = get_working_tape()
        tape.clear_tape()

        # solve forward
        for _ in range(self.tt_steps):

            # Create Functions for the solution and time-lagged solution
            self.u_.project(self.u)

            solve(self._form == 0, self.u)

            # Store forward solution at exports so we can plot again later
            self._solutions.append(self.u.copy(deepcopy=True))
            
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
        return inner(sol, sol) * ds(2)

    return qoi


# Initial mesh for all test cases
initial_mesh = UnitSquareMesh(30, 30)
