from firedrake import *
from firedrake.petsc import PETSc
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
    steps = 20

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

    def __init__(self, meshes, ic, **kwargs):
        """
        :arg mesh: the mesh to define the solver on
        :arg ic: the current state / initial condition
        """
        self.meshes = meshes

        # Collect parameters
        self._step = 0
        self.total_steps = parameters.steps
        self.dt = Constant(parameters.timestep)
        assert len(meshes) == self.total_steps
        
        self.nu = [parameters.viscosity(meshes[i]) for i in range(self.total_steps)]

        # Define variational formulation
        V = self.function_space
        self.u = [Function(V[i]) for i in range(self.total_steps)]
        self.u_ = [Function(V[i]) for i in range(self.total_steps)]
        v = [TestFunction(V[i]) for i in range(self.total_steps)]
        self._form = [(
            inner((self.u[i] - self.u_[i]) / self.dt, v[i]) * dx
            + inner(dot(self.u[i], nabla_grad(self.u[i])), v[i]) * dx
            + self.nu[i] * inner(grad(self.u[i]), grad(v[i])) * dx
        ) for i in range(self.total_steps)]

        # Set initial condition
        self.u_[0].project(parameters.ic(meshes[0]))

    @property
    def function_space(self):
        r"""
        The :math:`\mathbb P2` finite element space.
        """
        return [get_function_space(self.meshes[i]) for i in range(self.total_steps)]

    @property
    def form(self):
        """
        The weak form of Burgers equation
        """
        return self._form

    @property
    def solution(self):
        return self.u
    
    @property
    def step(self):
        return self._step

    def iterate(self, **kwargs):
        """
        Take a single timestep of Burgers equation
        """
        for i in range(self.total_steps):
            print(f"iter {self._step}")
            
            # Set problem
            problem = NonlinearVariationalProblem(self._form[self._step], self.u[self._step])

            # Create solver
            self._solver = NonlinearVariationalSolver(problem)
            self._solver.solve()
            
            # Set timestep
            self._step += 1
            
            if self._step+1 < self.total_steps:
                self.u_[self._step+1].assign(self.u[self._step])


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
initial_mesh = [UnitSquareMesh(30, 30) for _ in range(20)]



# A simple pretest
a = Solver(meshes = initial_mesh, ic = 0, kwargs='0')
a.iterate()
b = a.solution
import matplotlib.pyplot as plt
# fig, axes = plt.subplots(20)
# for i in range(20):
#     tricontourf(b[i], axes=axes[i])
fig, axes = plt.subplots()
tricontourf(b[10], axes=axes)
plt.show()
