from thetis import *
import numpy as np


class Parameters(object):
    h_min = 1.0e-05
    h_max = 500.0

    viscosity = Constant(0.5)
    depth = 40.0
    drag_coefficient = Constant(0.0025)

    turbine_diameter = 18.0
    turbine_coords = []
    thrust_coefficient = 0.8
    density = Constant(1030.0*1.0e-06)

    solver_parameters = {
        'mat_type': 'aij',
        'snes_type': 'newtonls',
        'snes_linesearch_type': 'bt',
        'snes_rtol': 1.0e-08,
        'snes_max_it': 100,
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
    }
    adjoint_solver_parameters = solver_parameters

    @property
    def num_turbines(self):
        return len(self.turbine_coords)

    @property
    def turbine_ids(self):
        return list(2 + np.arange(self.num_turbines, dtype=np.int32))

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

    def bathymetry(self, mesh):
        P0_2d = get_functionspace(mesh, 'DG', 0)
        return Function(P0_2d).assign(parameters.depth)

    def u_inflow(self, mesh):
        return as_vector([5, 0])

    def Re(self, fwd_sol):
        u = fwd_sol.split()[0]
        unorm = sqrt(dot(u, u))
        mesh = u.function_space().mesh()
        P0 = get_functionspace(mesh, 'DG', 0)
        h = CellSize(mesh)
        return interpolate(0.5*h*unorm/self.viscosity, P0)

    def turbine_density(self, mesh):
        return Constant(1.0/self.turbine_area, domain=mesh)

    def farm(self, mesh):
        farm_options = TidalTurbineFarmOptions()
        farm_options.turbine_density = self.turbine_density(mesh)
        farm_options.turbine_options.diameter = self.turbine_diameter
        farm_options.turbine_options.thrust_coefficient = self.corrected_thrust_coefficient
        return {farm_id: farm_options for farm_id in self.turbine_ids}

    def drag(self, mesh):
        P0 = FunctionSpace(mesh, 'DG', 0)
        p0test = TestFunction(P0)
        ret = Function(P0)

        # Background drag
        Cb = self.drag_coefficient
        expr = p0test*Cb*dx(domain=mesh)

        # Turbine drag
        Ct = self.corrected_thrust_coefficient
        Cd = 0.5*Ct*self.turbine_area*self.turbine_density(mesh)
        for tag in self.turbine_ids:
            expr += p0test*Cd*dx(tag, domain=mesh)

        assemble(expr, tensor=ret)
        return ret


PETSc.Sys.popErrorHandler()
parameters = Parameters()


def get_function_space(mesh):
    return get_functionspace(mesh, 'DG', 1, vector=True)*get_functionspace(mesh, 'CG', 2)


def setup_solver(mesh, ic):
    bathymetry = parameters.bathymetry(mesh)
    Cd = parameters.drag_coefficient

    # Create solver object
    solver_obj = solver2d.FlowSolver2d(mesh, bathymetry)
    options = solver_obj.options
    options.element_family = 'dg-cg'
    options.timestep = 20.0
    options.simulation_export_time = 20.0
    options.simulation_end_time = 18.0
    options.swe_timestepper_type = 'SteadyState'
    options.swe_timestepper_options.solver_parameters = parameters.solver_parameters
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
    options.tidal_turbine_farms = parameters.farm(mesh)

    # Apply initial guess
    u_init, eta_init = ic.split()
    solver_obj.assign_initial_conditions(uv=u_init, elev=eta_init)
    return solver_obj


def get_initial_condition(function_space):
    q = Function(function_space)
    u, eta = q.split()
    u.interpolate(parameters.u_inflow(function_space.mesh()))
    return q


def get_qoi(mesh):
    rho = parameters.density
    At = parameters.turbine_area
    Aswept = parameters.swept_area
    Ct = parameters.thrust_coefficient
    ct = Constant(0.5*Aswept*Ct/At)

    def qoi(sol):
        u, eta = split(sol)
        J = rho*ct*sum([
            pow(dot(u, u), 1.5)*dx(tag)
            for tag in parameters.turbine_ids
        ])
        return J

    return qoi


def get_bnd_functions(n, eta_in, u_in, bnd_marker, bnd_conditions):
    funcs = bnd_conditions[bnd_marker]
    eta_ext = u_ext = None
    if 'elev' in funcs and 'uv' in funcs:
        eta_ext = funcs['elev']
        u_ext = funcs['uv']
    elif 'elev' in funcs and 'un' in funcs:
        eta_ext = funcs['elev']
        u_ext = funcs['un']*n
    elif 'elev' in funcs:
        eta_ext = funcs['elev']
        u_ext = u_in
    elif 'uv' in funcs:
        eta_ext = eta_in
        u_ext = funcs['uv']
    elif 'un' in funcs:
        eta_ext = eta_in
        u_ext = funcs['un']*n
    return eta_ext, u_ext


def dwr_indicator(mesh, q, q_star):
    mesh_plus = q.function_space().mesh()

    # Extract parameters from solver object
    solver_obj = setup_solver(mesh_plus, q)
    options = solver_obj.options
    b = solver_obj.fields.bathymetry_2d
    g = physical_constants['g_grav']
    nu = options.horizontal_viscosity
    eta_is_dg = options.element_family == 'dg-dg'
    u, eta = split(q)
    u_old, eta_old = u, eta  # NOTE: hard-coded for steady-state case
    z, zeta = split(q_star)
    H = eta_old + b
    alpha = options.sipg_factor
    cell = mesh_plus.ufl_cell()
    p = options.polynomial_degree
    cp = (p + 1)*(p + 2)/2 if cell == triangle else (p + 1)**2
    l_normal = CellVolume(mesh_plus)/FacetArea(mesh_plus)
    sigma = alpha*cp/l_normal
    sp = sigma('+')
    sm = sigma('-')
    sigma = conditional(sp > sm, sp, sm)
    f = options.coriolis_frequency
    C_D = options.quadratic_drag_coefficient

    P0_plus = get_functionspace(mesh_plus, 'DG', 0)
    p0test = TestFunction(P0_plus)
    p0trial = TrialFunction(P0_plus)
    n = FacetNormal(mesh_plus)

    def restrict(v):
        try:
            return jump(v, p0test)
        except Exception:
            return v('+')*p0test('+') + v('-')*p0test('-')

    # --- Element residual

    R = 0
    if eta_is_dg:
        R += p0test*g*nabla_div(z)*eta*dx
    else:
        R += -p0test*g*inner(z, grad(eta))*dx
    R += -p0test*zeta*div(H*u)*dx
    R += -p0test*inner(z, dot(u_old, nabla_grad(u)))*dx
    if options.use_grad_div_viscosity_term:
        stress = 2.0*nu*sym(grad(u))
    else:
        stress = nu*grad(u)
    R += p0test*inner(z, div(stress))*dx
    if options.use_grad_depth_viscosity_term:
        R += p0test*inner(z, dot(grad(H)/H))*dx
    if f is not None:
        R += -p0test*f*(-u[1]*z[0] + u[0]*z[1])*dx
    unorm = sqrt(dot(u, u))
    if C_D is not None:
        R += -p0test*C_D*unorm*inner(z, u)/H*dx
    unorm = sqrt(dot(u_old, u_old))
    for subdomain_id, farm_options in options.tidal_turbine_farms.items():
        density = farm_options.turbine_density
        C_T = farm_options.turbine_options.thrust_coefficient
        A_T = pi*(0.5*farm_options.turbine_options.diameter)**2
        C_D = 0.5*C_T*A_T*density
        R += -p0test*C_D*unorm*inner(z, u)/H*dx(subdomain_id)

    # --- Inter-element flux

    r = 0
    r += restrict(inner(dot(H*u, n), zeta))*dS
    if eta_is_dg:
        h = avg(H)
        head_star = avg(eta) + sqrt(h/g)*jump(u, n)
        r += -head_star*restrict(g*dot(z, n))*dS
        r += restrict(g*eta*dot(z, n))*dS

        u_rie = avg(u) + sqrt(g/h)*jump(eta, n)
        r += -inner(h*u_rie, restrict(zeta*n))*dS

    r += -inner(jump(u, n)*avg(u), restrict(z))*dS
    u_lax_friedrichs = options.lax_friedrichs_velocity_scaling_factor
    if options.use_lax_friedrichs_velocity:
        gamma = 0.5*abs(dot(avg(u_old), n('-')))*u_lax_friedrichs
        r += -inner(gamma*jump(u), restrict(z))*dS

    if options.use_grad_div_viscosity_term:
        stress_jump = 2.0*avg(nu)*jump(sym(outer(u, n)))
    else:
        stress_jump = avg(nu)*jump(outer(u, n))
    r += -inner(avg(dot(stress, n)), restrict(z))*dS

    r += -inner(sigma*avg(nu)*jump(outer(u, n)), restrict(outer(z, n)))*dS
    r += 0.5*inner(stress_jump, restrict(grad(z)))*dS

    # --- Boundary flux

    bnd_markers = [1, 2, 3]  # NOTE: hard-coded
    bnd_conditions = solver_obj.bnd_functions['shallow_water']
    r += p0test*inner(H*dot(u, n), zeta)*ds  # NOTE: assumes freeslip on whole boundary
    r += p0test*dot(u, n)*inner(u, z)*ds
    for bnd_marker in bnd_markers:
        funcs = bnd_conditions.get(bnd_marker)
        ds_bnd = ds(int(bnd_marker))

        if eta_is_dg:
            r += p0test*inner(g*eta*n, z)*ds
            if funcs is not None:
                eta_ext, u_ext = get_bnd_functions(n, eta, u, bnd_marker, bnd_conditions)
                un_jump = inner(u - u_ext, n)
                eta_rie = 0.5*(eta + eta_ext) + sqrt(H/g)*un_jump
                r += -p0test*inner(g*eta_rie*n, z)*ds_bnd
            if funcs is None or 'symm' in funcs:
                head_rie = eta + sqrt(H/g)*inner(u, n)
                r += -p0test*inner(g*head_rie*n, z)*ds_bnd
        else:
            if funcs is not None:
                eta_ext, u_ext = get_bnd_functions(n, eta, u, bnd_marker, bnd_conditions)
                un_jump = inner(u - u_ext, n)
                eta_rie = 0.5*(eta + eta_ext) + sqrt(H/g)*un_jump
                r += -p0test*inner(g*(eta_rie-eta)*n, z)*ds_bnd

        if funcs is None:
            eta_ext, u_ext = get_bnd_functions(n, eta, u, bnd_marker, bnd_conditions)
            eta_ext_old, u_ext_old = get_bnd_functions(n, eta_old, u_old, bnd_marker, bnd_conditions)
            H_ext = eta_ext_old + b
            h_av = 0.5*(H + H_ext)
            eta_jump = eta - eta_ext
            un_rie = 0.5*inner(u + u_ext, n) + sqrt(g/h_av)*eta_jump
            un_jump = inner(u_old - u_ext_old, n)
            eta_rie = 0.5*(eta_old + eta_ext_old) + sqrt(h_av/g)*un_jump
            h_rie = b + eta_rie
            r += -p0test*inner(h_rie*un_rie, zeta)*ds_bnd

            un_av = dot(avg(u_old), n('-'))
            eta_jump = eta_old - eta_ext_old
            un_rie = 0.5*inner(u_old + u_ext_old, n) + sqrt(g/H)*eta_jump
            r += -p0test*un_rie*dot(0.5*(u_ext + u), z)*ds_bnd
            if options.use_lax_friedrichs_velocity:
                gamma = 0.5*abs(un_av)*u_lax_friedrichs
                u_ext = u - 2*dot(u, n)*n
                gamma = 0.5*abs(dot(u_old, n))*u_lax_friedrichs
                r += -p0test*gamma*dot(u - u_ext, z)*ds_bnd

            if 'un' in funcs:
                delta_u = (dot(u, n) - funcs['un'])*n
            else:
                eta_ext, u_ext = get_bnd_functions(n, eta, u, bnd_marker, bnd_conditions)
                if u_ext is u:
                    continue
                delta_u = u - u_ext

            if options.use_grad_div_viscosity_term:
                stress_jump = 2.0*nu*sym(outer(delta_u, n))
            else:
                stress_jump = nu*outer(delta_u, n)

            r += -p0test*sigma*inner(nu*delta_u, z)*ds_bnd
            r += p0test*inner(stress_jump, grad(z))*ds_bnd

    # Process R and r
    residual = Function(P0_plus).assign(assemble(R))
    sp = {
        'mat_type': 'matfree',
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'pc_type': 'jacobi',
    }
    flux = Function(P0_plus)
    solve(p0test*p0trial*dx == r, flux, solver_parameters=sp)
    dwr_plus = Function(P0_plus).assign(residual + flux)

    # Project down to base space
    P0 = get_functionspace(mesh, 'DG', 0)
    dwr = project(dwr_plus, P0)
    dwr.interpolate(abs(dwr))
    return dwr, dwr_plus
