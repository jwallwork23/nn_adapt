from thetis import *
from pyroteus import *
import pyroteus.go_mesh_seq
import nn_adapt.model

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
    
    # time dependent parameters
    end_time = 100.
    time_steps = 10.

    # Physical parameters
    viscosity_coefficient = 0.5
    depth = 40.0
    drag_coefficient = Constant(0.0025)
    inflow_speed = 5.0
    density = Constant(1030.0 * 1.0e-06)

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

    def ic(self, mesh):
        """
        Initial condition.
        """
        return self.u_inflow(mesh)

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


PETSc.Sys.popErrorHandler()
parameters = Parameters()
    
def GoalOrientedMeshSeq(mesh):
    
    fields = ["q"]
    time_partition = TimeInterval(parameters.end_time, 
                                        parameters.time_steps, 
                                        fields, timesteps_per_export=1)


    def get_solver(mesh_seq):

        def solver(index, ic):
            V = mesh_seq.function_spaces["q"][index]
            q = ic["q"]
            mesh_seq.form(index, {"q": (q, q)})
            u_init, eta_init = q.split()
            mesh_seq._thetis_solver.assign_initial_conditions(uv=u_init, elev=eta_init)
            mesh_seq._thetis_solver.iterate()
            
            return {"q": mesh_seq._thetis_solver.fields.solution_2d}

        return solver


    def get_form(mesh_seq):
        def form(index, ic):
            P = mesh_seq.time_partition
            
            bathymetry = parameters.bathymetry(mesh_seq[index])
            Cd = parameters.drag_coefficient

            # Create solver object
            mesh_seq._thetis_solver = solver2d.FlowSolver2d(mesh_seq[index], bathymetry)
            options = mesh_seq._thetis_solver.options
            options.element_family = "dg-cg"
            options.timestep = P.timestep
            options.simulation_export_time = P.timestep * P.timesteps_per_export[index]
            options.simulation_end_time = P.end_time
            options.swe_timestepper_type = "SteadyState"
            options.swe_timestepper_options.solver_parameters = parameters.solver_parameters
            options.use_grad_div_viscosity_term = False
            options.horizontal_viscosity = parameters.viscosity(mesh_seq[index])
            options.quadratic_drag_coefficient = Cd
            options.use_lax_friedrichs_velocity = True
            options.lax_friedrichs_velocity_scaling_factor = Constant(1.0)
            options.use_grad_depth_viscosity_term = False
            options.no_exports = True

            # Apply boundary conditions
            mesh_seq._thetis_solver.create_function_spaces()
            P1v_2d = mesh_seq._thetis_solver.function_spaces.P1v_2d
            u_inflow = interpolate(parameters.u_inflow(mesh_seq[index]), P1v_2d)
            mesh_seq._thetis_solver.bnd_functions["shallow_water"] = {
                1: {"uv": u_inflow},  # inflow
                2: {"elev": Constant(0.0)},  # outflow
                3: {"un": Constant(0.0)},  # free-slip
                4: {"uv": Constant(as_vector([0.0, 0.0]))},  # no-slip
                5: {"elev": Constant(0.0), "un": Constant(0.0)}  # weakly reflective
            }

            # # Create tidal farm
            options.tidal_turbine_farms = parameters.farm(mesh_seq[index])
            mesh_seq._thetis_solver.create_timestepper()
            
            return mesh_seq._thetis_solver.timestepper.F

        return form


    def get_function_space(mesh):
        """
        Construct the (mixed) finite element space used for the
        prognostic solution.
        """
        P1v_2d = get_functionspace(mesh, "DG", 1, vector=True)
        P2_2d = get_functionspace(mesh, "CG", 2)
        return {"q": P1v_2d * P2_2d}


    def get_initial_condition(mesh_seq):
        """
        Compute an initial condition based on the inflow velocity
        and zero free surface elevation.
        """
        V = mesh_seq.function_spaces["q"][0]
        q = Function(V)
        u, eta = q.split()
        u.interpolate(parameters.ic(mesh_seq))
        return {"q": q}


    def get_qoi(mesh_seq, solutions, index):
        """
        Extract the quantity of interest function from the :class:`Parameters`
        object.

        It should have one argument - the prognostic solution.
        """
        dt = Constant(mesh_seq.time_partition[index].timestep)
        rho = parameters.density
        Ct = parameters.corrected_thrust_coefficient
        At = parameters.swept_area
        Cd = 0.5 * Ct * At * parameters.turbine_density(mesh_seq[index])
        tags = parameters.turbine_ids
        sol = solutions["q"]

        def qoi():
            u, eta = split(sol)
            return sum([rho * Cd * pow(dot(u, u), 1.5) * dx(tag) for tag in tags])

        return qoi
    
    
    mesh_seq = pyroteus.go_mesh_seq.GoalOrientedMeshSeq(
        time_partition,
        mesh,
        get_function_spaces=get_function_space,
        get_initial_condition=get_initial_condition,
        get_form=get_form,
        get_solver=get_solver,
        get_qoi=get_qoi,
        qoi_type="end_time")
    
    return mesh_seq
    
        
        

# print(solutions["q"].keys())

# fwd_file = File("./out/forward.pvd")
# for i in range(len(solutions["u"]["forward"][0])):
#     fwd_file.write(*solutions["q"]["forward"][0][i].split())
    
# adj_file = File("./out/adjoint.pvd")
# for i in range(len(solutions["u"]["adjoint"][0])):
#     adj_file.write(*solutions["q"]["adjoint"][0][i].split())

# adj_next_file = File("./out/adjoint_next.pvd")
# for i in range(len(solutions["u"]["adjoint_next"][0])):
#     adj_next_file.write(*solutions["q"]["adjoint_next"][0][i].split())

