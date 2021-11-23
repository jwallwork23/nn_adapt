from thetis import *
from pyroteus_adjoint import *


class Parameters(object):
    num_inputs = 23
    num_outputs = 3
    dofs_per_element = 3
    h_min = 1.0e-05
    h_max = 500.0

    viscosity = Constant(0.5)
    depth = 40.0
    drag_coefficient = Constant(0.0025)

    turbine_diameter = 18.0
    num_turbines = 1
    thrust_coefficient = 0.8
    density = Constant(1030.0)

    @property
    def turbine_ids(self):
        return list(2 + np.array(range(self.num_turbines)))

    @property
    def turbine_area(self):
        return pi*(0.5*self.turbine_diameter)**2

    @property
    def swept_area(self):
        return self.depth*self.turbine_diameter

    @property
    def corrected_thrust_coefficient(self):
        A = self.turbine_area
        depth = self.depth
        D = self.turbine_diameter
        Ct = self.thrust_coefficient
        return 4.0/(1.0 + sqrt(1.0 - A/(depth*D)))**2*Ct

    def u_inflow(self, mesh):
        return as_vector([5, 0])

    def Re(self, fwd_sol):
        u = fwd_sol.split()[0]
        unorm = sqrt(dot(u, u))
        mesh = u.function_space().mesh()
        P0 = FunctionSpace(mesh, 'DG', 0)
        h = CellSize(mesh)
        return interpolate(0.5*h*unorm/self.viscosity, P0)


PETSc.Sys.popErrorHandler()
parameters = Parameters()
fields = ['q']


def get_function_spaces(mesh):
    return {'q': VectorFunctionSpace(mesh, 'DG', 1)*get_functionspace(mesh, 'DG', 1)}


def setup_solver(mesh, ic):
    P1_2d = get_functionspace(mesh, 'CG', 1)

    # Extract test case specific parameters
    bathymetry = Function(P1_2d).assign(parameters.depth)
    Cd = parameters.drag_coefficient

    # Create solver object
    solver_obj = solver2d.FlowSolver2d(mesh, bathymetry)
    options = solver_obj.options
    options.timestep = 20.0
    options.simulation_export_time = 20.0
    options.simulation_end_time = 18.0
    options.swe_timestepper_type = 'SteadyState'
    options.swe_timestepper_options.solver_parameters = {
        'mat_type': 'aij',
        'snes_type': 'newtonls',
        'snes_linesearch_type': 'bt',
        'snes_rtol': 1.0e-08,
        'snes_max_it': 100,
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
    }
    options.use_grad_div_viscosity_term = False
    options.horizontal_viscosity = parameters.viscosity
    options.quadratic_drag_coefficient = Cd
    options.use_lax_friedrichs_velocity = True
    options.lax_friedrichs_velocity_scaling_factor = Constant(1.0)
    options.use_grad_depth_viscosity_term = False
    options.no_exports = True
    solver_obj.create_equations()

    # Apply boundary conditions
    P1v_2d = solver_obj.function_spaces.P1v_2d
    u_inflow = interpolate(parameters.u_inflow(mesh), P1v_2d)
    solver_obj.bnd_functions['shallow_water'] = {
        1: {'uv': u_inflow},
        2: {'elev': Constant(0.0)},
        3: {'un': Constant(0.0)},
    }

    # Create tidal farm
    thrust_coefficient = parameters.corrected_thrust_coefficient
    farm_options = TidalTurbineFarmOptions()
    area = parameters.turbine_area
    farm_options.turbine_density = Constant(1.0/area, domain=mesh)
    farm_options.turbine_options.diameter = parameters.turbine_diameter
    farm_options.turbine_options.thrust_coefficient = thrust_coefficient
    solver_obj.options.tidal_turbine_farms = {
        farm_id: farm_options
        for farm_id in parameters.turbine_ids
    }

    # Apply initial guess
    u_init, eta_init = ic['q'].split()
    solver_obj.assign_initial_conditions(uv=u_init, elev=eta_init)
    return solver_obj


def get_solver(mesh_seq):

    def solver(i, ic):
        mesh = mesh_seq[i]
        solver_obj = setup_solver(mesh, ic)
        solver_obj.iterate()
        return {'q': solver_obj.fields.solution_2d}

    return solver


def get_initial_condition(mesh_seq):
    fs = mesh_seq.function_spaces['q'][0]
    q = Function(fs)
    u, eta = q.split()
    u.interpolate(parameters.u_inflow(mesh_seq[0]))
    return {'q': q}


def get_qoi(mesh_seq, i):
    rho = parameters.density
    At = parameters.turbine_area
    Aswept = parameters.swept_area
    Ct = parameters.thrust_coefficient
    ct = Constant(0.5*Aswept*Ct/At)

    def qoi(sol):
        u, eta = sol['q'].split()
        J = rho*ct*sum([
            pow(dot(u, u), 1.5)*dx(tag)
            for tag in parameters.turbine_ids
        ])
        return J

    return qoi


def dwr_indicator(mesh_seq, q, q_star):
    mesh_plus = q.function_space().mesh()

    # Evaluate indicator in enriched space
    solver_obj = setup_solver(mesh_plus, {'q': q})
    F = solver_obj.timestepper.F
    V = solver_obj.function_spaces.V_2d
    dwr_plus = get_dwr_indicator(F, q_star, test_space=V)

    # Project down to base space
    P0 = FunctionSpace(mesh_seq[0], 'DG', 0)
    dwr = project(dwr_plus, P0)
    dwr.interpolate(abs(dwr))
    return dwr, dwr_plus
