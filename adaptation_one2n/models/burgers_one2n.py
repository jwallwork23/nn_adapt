from copy import deepcopy
from firedrake import *
from firedrake.petsc import PETSc
from firedrake_adjoint import *
from firedrake.adjoint import get_solve_blocks
import nn_adapt.model
import nn_adapt.solving
from thetis import *


'''
A memory hungry method solving time dependent PDE.
'''
class Parameters(nn_adapt.model.Parameters):
    """
    Class encapsulating all parameters required for the tidal
    farm modelling test case.
    """

    discrete = False

    qoi_name = "power output"
    qoi_unit = "MW"

    # Adaptation parameters
    h_min = 1.0e-08
    h_max = 500.0

    # time steps
    tt_steps = 10
    timestep = 0.1

    # Physical parameters
    viscosity_coefficient = 0.5
    depth = 40.0
    drag_coefficient = Constant(0.0025)
    inflow_speed = 5.0
    density = Constant(1030.0 * 1.0e-06)

    # Additional setup
    viscosity_coefficient = 0.0001
    initial_speed = 1.0

    # Turbine parameters
    turbine_diameter = 18.0
    turbine_width = None
    turbine_coords = []
    thrust_coefficient = 0.8
    correct_thrust = True

    # Solver parameters
    solver_parameters = {
        "mat_type": "aij",
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1.0e-08,
        "snes_max_it": 100,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    adjoint_solver_parameters = solver_parameters

    @property
    def num_turbines(self):
        """
        Count the number of turbines based on the number
        of coordinates.
        """
        return len(self.turbine_coords)

    @property
    def turbine_ids(self):
        """
        Generate the list of turbine IDs, i.e. cell tags used
        in the gmsh geometry file.
        """
        if self.discrete:
            return list(2 + np.arange(self.num_turbines, dtype=np.int32))
        else:
            return ["everywhere"]

    @property
    def footprint_area(self):
        """
        Calculate the area of the turbine footprint in the horizontal.
        """
        d = self.turbine_diameter
        w = self.turbine_width or d
        return d * w

    @property
    def swept_area(self):
        """
        Calculate the area swept by the turbine in the vertical.
        """
        return pi * (0.5 * self.turbine_diameter) ** 2

    @property
    def cross_sectional_area(self):
        """
        Calculate the cross-sectional area of the turbine footprint
        in the vertical.
        """
        return self.depth * self.turbine_diameter

    @property
    def corrected_thrust_coefficient(self):
        """
        Correct the thrust coefficient to account for the
        fact that we use the velocity at the turbine, rather
        than an upstream veloicity.

        See [Kramer and Piggott 2016] for details.
        """
        Ct = self.thrust_coefficient
        if not self.correct_thrust:
            return Ct
        At = self.swept_area
        corr = 4.0 / (1.0 + sqrt(1.0 - Ct * At / self.cross_sectional_area)) ** 2
        return Ct * corr

    def bathymetry(self, mesh):
        """
        Compute the bathymetry field on the current `mesh`.
        """
        # NOTE: We assume a constant bathymetry field
        P0_2d = get_functionspace(mesh, "DG", 0)
        return Function(P0_2d).assign(parameters.depth)

    def u_inflow(self, mesh):
        """
        Compute the inflow velocity based on the current `mesh`.
        """
        # NOTE: We assume a constant inflow
        return as_vector([self.inflow_speed, 0])

    # def ic(self, mesh):
    #     """
    #     Initial condition.
    #     """
    #     return self.u_inflow(mesh)
    def ic(self, mesh):
        """
        Initial condition
        """
        x, y = SpatialCoordinate(mesh)
        expr = self.initial_speed * sin(pi * x)
        yside = self.initial_speed * sin(pi * y)
        yside = 0
        return as_vector([expr, yside])

    def turbine_density(self, mesh):
        """
        Compute the turbine density function on the current `mesh`.
        """
        if self.discrete:
            return Constant(1.0 / self.footprint_area, domain=mesh)
        x, y = SpatialCoordinate(mesh)
        r2 = self.turbine_diameter / 2
        r1 = r2 if self.turbine_width is None else self.turbine_width / 2

        def bump(x0, y0, scale=1.0):
            qx = ((x - x0) / r1) ** 2
            qy = ((y - y0) / r2) ** 2
            cond = And(qx < 1, qy < 1)
            b = exp(1 - 1 / (1 - qx)) * exp(1 - 1 / (1 - qy))
            return conditional(cond, Constant(scale) * b, 0)

        bumps = 0
        for xy in self.turbine_coords:
            bumps += bump(*xy, scale=1 / assemble(bump(*xy) * dx))
        return bumps

    def farm(self, mesh):
        """
        Construct a dictionary of :class:`TidalTurbineFarmOptions`
        objects based on the current `mesh`.
        """
        Ct = self.corrected_thrust_coefficient
        farm_options = TidalTurbineFarmOptions()
        farm_options.turbine_density = self.turbine_density(mesh)
        farm_options.turbine_options.diameter = self.turbine_diameter
        farm_options.turbine_options.thrust_coefficient = Ct
        return {farm_id: farm_options for farm_id in self.turbine_ids}

    def turbine_drag(self, mesh):
        """
        Compute the contribution to the drag coefficient due to the
        tidal turbine parametrisation on the current `mesh`.
        """
        P0_2d = get_functionspace(mesh, "DG", 0)
        p0test = TestFunction(P0_2d)
        Ct = self.corrected_thrust_coefficient
        At = self.swept_area
        Cd = 0.5 * Ct * At * self.turbine_density(mesh)
        return sum([p0test * Cd * dx(tag, domain=mesh) for tag in self.turbine_ids])

    def drag(self, mesh, background=False):
        r"""
        Create a :math:`\mathbb P0` field for the drag on the current
        `mesh`.

        :kwarg background: should we consider the background drag
            alone, or should the turbine drag be included?
        """
        P0_2d = get_functionspace(mesh, "DG", 0)
        ret = Function(P0_2d)

        # Background drag
        Cb = self.drag_coefficient
        if background:
            return ret.assign(Cb)
        p0test = TestFunction(P0_2d)
        expr = p0test * Cb * dx(domain=mesh)

        # Turbine drag
        assemble(expr + self.turbine_drag(mesh), tensor=ret)
        return ret

    def viscosity(self, mesh):
        r"""
        Create a :math:`\mathbb P0` field for the viscosity coefficient
        on the current `mesh`.
        """
        # NOTE: We assume a constant viscosity coefficient
        P0_2d = get_functionspace(mesh, "DG", 0)
        return Function(P0_2d).assign(self.viscosity_coefficient)


# class Parameters(nn_adapt.model.Parameters):
#     """
#     Class encapsulating all parameters required for a simple
#     Burgers equation test case.
#     """

#     qoi_name = "right boundary integral"
#     qoi_unit = r"m\,s^{-1}"

#     # Adaptation parameters
#     h_min = 1.0e-10  # Minimum metric magnitude
#     h_max = 1.0  # Maximum metric magnitude

#     # Physical parameters
#     viscosity_coefficient = 0.0001
#     initial_speed = 1.0

#     # Timestepping parameters
#     timestep = 0.05
#     tt_steps = 10

#     solver_parameters = {}
#     adjoint_solver_parameters = {}

#     def bathymetry(self, mesh):
#         """
#         Compute the bathymetry field on the current `mesh`.

#         Note that there isn't really a concept of bathymetry
#         for Burgers equation. It is kept constant and should
#         be ignored by the network.
#         """
#         P0_2d = FunctionSpace(mesh, "DG", 0)
#         return Function(P0_2d).assign(1.0)

#     def drag(self, mesh):
#         """
#         Compute the bathymetry field on the current `mesh`.

#         Note that there isn't really a concept of bathymetry
#         for Burgers equation. It is kept constant and should
#         be ignored by the network.
#         """
#         P0_2d = FunctionSpace(mesh, "DG", 0)
#         return Function(P0_2d).assign(1.0)

#     def viscosity(self, mesh):
#         """
#         Compute the viscosity coefficient on the current `mesh`.
#         """
#         P0_2d = FunctionSpace(mesh, "DG", 0)
#         return Function(P0_2d).assign(self.viscosity_coefficient)

#     def ic(self, mesh):
#         """
#         Initial condition
#         """
#         x, y = SpatialCoordinate(mesh)
#         expr = self.initial_speed * sin(pi * x)
#         yside = self.initial_speed * sin(pi * y)
#         yside = 0
#         return as_vector([expr, yside])


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


# # Initial mesh for all test cases
# initial_mesh = UnitSquareMesh(30, 30)
