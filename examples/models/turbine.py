from thetis import *
import numpy as np


class Parameters(object):
    """
    Class encapsulating all parameters required for the tidal
    farm modelling test case.
    """

    qoi_unit = "MW"
    qoi_name = "power output"

    # Adaptation parameters
    h_min = 1.0e-05
    h_max = 500.0

    # Physical parameters
    viscosity = Constant(0.5)
    depth = 40.0
    drag_coefficient = Constant(0.0025)
    inflow_speed = 5.0
    density = Constant(1030.0 * 1.0e-06)

    # Turbine parameters
    turbine_diameter = 18.0
    turbine_coords = []
    thrust_coefficient = 0.8

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
        return list(2 + np.arange(self.num_turbines, dtype=np.int32))

    @property
    def footprint_area(self):
        """
        Calculate the area of the turbine footprint in the horizontal.
        """
        return self.turbine_diameter**2

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
        At = self.swept_area
        Ct = self.thrust_coefficient
        corr = 4.0 / (1.0 + sqrt(1.0 - Ct * At / self.cross_sectional_area)) ** 2
        return Ct * corr

    def bathymetry(self, mesh):
        """
        Compute the bathymetry field on the current mesh.
        """
        # NOTE: We assume a constant bathymetry field
        P0_2d = get_functionspace(mesh, "DG", 0)
        return Function(P0_2d).assign(parameters.depth)

    def u_inflow(self, mesh):
        """
        Compute the inflow velocity based on the current mesh.
        """
        # NOTE: We assume a constant inflow
        return as_vector([self.inflow_speed, 0])

    def Re(self, fwd_sol):
        """
        Compute the mesh Reynolds number based on the current
        forward solution.
        """
        u = fwd_sol.split()[0]
        unorm = sqrt(dot(u, u))
        mesh = u.function_space().mesh()
        P0 = get_functionspace(mesh, "DG", 0)
        h = CellSize(mesh)
        return interpolate(0.5 * h * unorm / self.viscosity, P0)

    def turbine_density(self, mesh):
        """
        Compute the turbine density function on the current mesh.
        """
        return Constant(1.0 / self.footprint_area, domain=mesh)

    def farm(self, mesh):
        """
        Construct a dictionary of :class:`TidalTurbineFarmOptions`
        objects based on the current mesh.
        """
        Ct = self.corrected_thrust_coefficient
        farm_options = TidalTurbineFarmOptions()
        farm_options.turbine_density = self.turbine_density(mesh)
        farm_options.turbine_options.diameter = self.turbine_diameter
        farm_options.turbine_options.thrust_coefficient = Ct
        return {farm_id: farm_options for farm_id in self.turbine_ids}

    def drag(self, mesh, background=False):
        r"""
        Create a :math:`\mathbb P0` field for the drag on the current
        mesh.

        :arg mesh: the current mesh
        :kwarg background: should we consider the background drag
            alone, or should the turbine drag be included?
        """
        P0 = FunctionSpace(mesh, "DG", 0)
        ret = Function(P0)

        # Background drag
        Cb = self.drag_coefficient
        if background:
            return ret.assign(Cb)
        p0test = TestFunction(P0)
        expr = p0test * Cb * dx(domain=mesh)

        # Turbine drag
        Ct = self.corrected_thrust_coefficient
        At = self.swept_area
        Cd = 0.5 * Ct * At * self.turbine_density(mesh)
        for tag in self.turbine_ids:
            expr += p0test * Cd * dx(tag, domain=mesh)
        assemble(expr, tensor=ret)
        return ret


PETSc.Sys.popErrorHandler()
parameters = Parameters()


def get_function_space(mesh):
    """
    Construct the (mixed) finite element space used for the
    prognostic solution.
    """
    return get_functionspace(mesh, "DG", 1, vector=True) * get_functionspace(
        mesh, "CG", 2
    )


def setup_solver(mesh, ic):
    """
    Set up the Thetis :class:`FlowSolver2d` object, based on
    the current mesh and initial condition.
    """
    bathymetry = parameters.bathymetry(mesh)
    Cd = parameters.drag_coefficient

    # Create solver object
    solver_obj = solver2d.FlowSolver2d(mesh, bathymetry)
    options = solver_obj.options
    options.element_family = "dg-cg"
    options.timestep = 20.0
    options.simulation_export_time = 20.0
    options.simulation_end_time = 18.0
    options.swe_timestepper_type = "SteadyState"
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
    solver_obj.bnd_functions["shallow_water"] = {
        1: {"uv": u_inflow},
        2: {"elev": Constant(0.0)},
        3: {"un": Constant(0.0)},
    }

    # Create tidal farm
    options.tidal_turbine_farms = parameters.farm(mesh)

    # Apply initial guess
    u_init, eta_init = ic.split()
    solver_obj.assign_initial_conditions(uv=u_init, elev=eta_init)
    return solver_obj


def get_initial_condition(function_space):
    """
    Compute an initial condition based on the inflow velocity
    and zero free surface elevation.
    """
    q = Function(function_space)
    u, eta = q.split()
    u.interpolate(parameters.u_inflow(function_space.mesh()))
    return q


def get_qoi(mesh):
    """
    Extract the quantity of interest function from the :class:`Parameters`
    object.

    It should have one argument - the prognostic solution.
    """
    rho = parameters.density
    At = parameters.swept_area
    A = parameters.footprint_area
    Ct = parameters.thrust_coefficient
    ct = Constant(0.5 * Ct * At / A)
    tags = parameters.turbine_ids

    def qoi(sol):
        u, eta = split(sol)
        return sum([rho * ct * pow(dot(u, u), 1.5) * dx(tag) for tag in tags])

    return qoi


def dwr_indicator(mesh, q, q_star):
    r"""
    Evaluate the DWR error indicator as a :math:`\mathbb P0` field.

    :arg mesh: the current mesh
    :arg q: the forward solution, transferred into enriched space
    :arg q_star: the adjoint solution in enriched space
    """
    mesh_plus = q.function_space().mesh()

    # Extract parameters from solver object
    solver_obj = setup_solver(mesh_plus, q)
    options = solver_obj.options
    bnd_conditions = solver_obj.bnd_functions["shallow_water"]
    b = solver_obj.fields.bathymetry_2d
    g = physical_constants["g_grav"]
    nu = options.horizontal_viscosity
    eta_is_dg = options.element_family == "dg-dg"
    u, eta = split(q)
    u_old, eta_old = u, eta  # NOTE: hard-coded for steady-state case
    z, zeta = split(q_star)
    H = eta_old + b
    alpha = options.sipg_factor
    cell = mesh_plus.ufl_cell()
    p = options.polynomial_degree
    cp = (p + 1) * (p + 2) / 2 if cell == triangle else (p + 1) ** 2
    l_normal = CellVolume(mesh_plus) / FacetArea(mesh_plus)
    sigma = alpha * cp / l_normal
    sp = sigma("+")
    sm = sigma("-")
    sigma = conditional(sp > sm, sp, sm)
    f = options.coriolis_frequency
    C_D = options.quadratic_drag_coefficient

    P0_plus = get_functionspace(mesh_plus, "DG", 0)
    p0test = TestFunction(P0_plus)
    p0trial = TrialFunction(P0_plus)
    n = FacetNormal(mesh_plus)

    def restrict(v):
        try:
            return jump(v, p0test)
        except Exception:
            return v("+") * p0test("+") + v("-") * p0test("-")

    def get_bnd_functions(eta_in, u_in, bnd_marker):
        funcs = bnd_conditions[bnd_marker]
        eta_ext = u_ext = None
        if "elev" in funcs and "uv" in funcs:
            eta_ext = funcs["elev"]
            u_ext = funcs["uv"]
        elif "elev" in funcs and "un" in funcs:
            eta_ext = funcs["elev"]
            u_ext = funcs["un"] * n
        elif "elev" in funcs:
            eta_ext = funcs["elev"]
            u_ext = u_in
        elif "uv" in funcs:
            eta_ext = eta_in
            u_ext = funcs["uv"]
        elif "un" in funcs:
            eta_ext = eta_in
            u_ext = funcs["un"] * n
        return eta_ext, u_ext

    # --- Element residual

    R = 0
    if eta_is_dg:
        R += p0test * g * nabla_div(z) * eta * dx
    else:
        R += -p0test * g * inner(z, grad(eta)) * dx
    R += -p0test * zeta * div(H * u) * dx
    R += -p0test * inner(z, dot(u_old, nabla_grad(u))) * dx
    if options.use_grad_div_viscosity_term:
        stress = 2.0 * nu * sym(grad(u))
    else:
        stress = nu * grad(u)
    R += p0test * inner(z, div(stress)) * dx
    if options.use_grad_depth_viscosity_term:
        R += p0test * inner(z, dot(grad(H) / H)) * dx
    if f is not None:
        R += -p0test * f * (-u[1] * z[0] + u[0] * z[1]) * dx
    unorm = sqrt(dot(u, u))
    if C_D is not None:
        R += -p0test * C_D * unorm * inner(z, u) / H * dx
    unorm = sqrt(dot(u_old, u_old))
    for subdomain_id, farm_options in options.tidal_turbine_farms.items():
        density = farm_options.turbine_density
        C_T = farm_options.turbine_options.thrust_coefficient
        A_T = pi * (0.5 * farm_options.turbine_options.diameter) ** 2
        C_D = 0.5 * C_T * A_T * density
        R += -p0test * C_D * unorm * inner(z, u) / H * dx(subdomain_id)

    # --- Inter-element flux

    r = 0
    r += restrict(inner(dot(H * u, n), zeta)) * dS
    if eta_is_dg:
        h = avg(H)
        head_star = avg(eta) + sqrt(h / g) * jump(u, n)
        r += -head_star * restrict(g * dot(z, n)) * dS
        r += restrict(g * eta * dot(z, n)) * dS

        u_rie = avg(u) + sqrt(g / h) * jump(eta, n)
        r += -inner(h * u_rie, restrict(zeta * n)) * dS

    r += -inner(jump(u, n) * avg(u), restrict(z)) * dS
    u_lax_friedrichs = options.lax_friedrichs_velocity_scaling_factor
    if options.use_lax_friedrichs_velocity:
        gamma = 0.5 * abs(dot(avg(u_old), n("-"))) * u_lax_friedrichs
        r += -inner(gamma * jump(u), restrict(z)) * dS

    if options.use_grad_div_viscosity_term:
        stress_jump = 2.0 * avg(nu) * jump(sym(outer(u, n)))
    else:
        stress_jump = avg(nu) * jump(outer(u, n))
    r += -inner(avg(dot(stress, n)), restrict(z)) * dS

    r += -inner(sigma * avg(nu) * jump(outer(u, n)), restrict(outer(z, n))) * dS
    r += 0.5 * inner(stress_jump, restrict(grad(z))) * dS

    # --- Boundary flux

    bnd_markers = [1, 2, 3]  # NOTE: hard-coded
    r += (
        p0test * inner(H * dot(u, n), zeta) * ds
    )  # NOTE: assumes freeslip on whole boundary
    r += p0test * dot(u, n) * inner(u, z) * ds
    for bnd_marker in bnd_markers:
        funcs = bnd_conditions.get(bnd_marker)
        ds_bnd = ds(int(bnd_marker))

        if eta_is_dg:
            r += p0test * inner(g * eta * n, z) * ds
            if funcs is not None:
                eta_ext, u_ext = get_bnd_functions(eta, u, bnd_marker)
                un_jump = inner(u - u_ext, n)
                eta_rie = 0.5 * (eta + eta_ext) + sqrt(H / g) * un_jump
                r += -p0test * inner(g * eta_rie * n, z) * ds_bnd
            if funcs is None or "symm" in funcs:
                head_rie = eta + sqrt(H / g) * inner(u, n)
                r += -p0test * inner(g * head_rie * n, z) * ds_bnd
        else:
            if funcs is not None:
                eta_ext, u_ext = get_bnd_functions(eta, u, bnd_marker)
                un_jump = inner(u - u_ext, n)
                eta_rie = 0.5 * (eta + eta_ext) + sqrt(H / g) * un_jump
                r += -p0test * inner(g * (eta_rie - eta) * n, z) * ds_bnd

        if funcs is None:
            eta_ext, u_ext = get_bnd_functions(eta, u, bnd_marker, bnd_conditions)
            eta_ext_old, u_ext_old = get_bnd_functions(eta_old, u_old, bnd_marker)
            H_ext = eta_ext_old + b
            h_av = 0.5 * (H + H_ext)
            eta_jump = eta - eta_ext
            un_rie = 0.5 * inner(u + u_ext, n) + sqrt(g / h_av) * eta_jump
            un_jump = inner(u_old - u_ext_old, n)
            eta_rie = 0.5 * (eta_old + eta_ext_old) + sqrt(h_av / g) * un_jump
            h_rie = b + eta_rie
            r += -p0test * inner(h_rie * un_rie, zeta) * ds_bnd

            un_av = dot(avg(u_old), n("-"))
            eta_jump = eta_old - eta_ext_old
            un_rie = 0.5 * inner(u_old + u_ext_old, n) + sqrt(g / H) * eta_jump
            r += -p0test * un_rie * dot(0.5 * (u_ext + u), z) * ds_bnd
            if options.use_lax_friedrichs_velocity:
                gamma = 0.5 * abs(un_av) * u_lax_friedrichs
                u_ext = u - 2 * dot(u, n) * n
                gamma = 0.5 * abs(dot(u_old, n)) * u_lax_friedrichs
                r += -p0test * gamma * dot(u - u_ext, z) * ds_bnd

            if "un" in funcs:
                delta_u = (dot(u, n) - funcs["un"]) * n
            else:
                eta_ext, u_ext = get_bnd_functions(eta, u, bnd_marker)
                if u_ext is u:
                    continue
                delta_u = u - u_ext

            if options.use_grad_div_viscosity_term:
                stress_jump = 2.0 * nu * sym(outer(delta_u, n))
            else:
                stress_jump = nu * outer(delta_u, n)

            r += -p0test * sigma * inner(nu * delta_u, z) * ds_bnd
            r += p0test * inner(stress_jump, grad(z)) * ds_bnd

    # Process R and r
    residual = Function(P0_plus).assign(assemble(R))
    sp = {
        "mat_type": "matfree",
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "jacobi",
    }
    flux = Function(P0_plus)
    solve(p0test * p0trial * dx == r, flux, solver_parameters=sp)
    dwr_plus = Function(P0_plus).assign(residual + flux)

    # Project down to base space
    P0 = get_functionspace(mesh, "DG", 0)
    dwr = project(dwr_plus, P0)
    dwr.interpolate(abs(dwr))
    return dwr
