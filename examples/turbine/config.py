from models.turbine import *


testing_cases = ["aligned", "offset"]


def l2dist(xy, xyt):
    r"""
    Usual :math:`\ell_2` distance between
    two points in Euclidean space.
    """
    diff = np.array(xy) - np.array(xyt)
    return np.sqrt(np.dot(diff, diff))


def sample_uniform(l, u):
    """
    Sample from the continuous uniform
    distribution :math:`U(l, u)`.

    :arg l: the lower bound
    :arg u: the upper bound
    """
    return l + (u - l) * np.random.rand()


def initialise(case, discrete=False):
    """
    Given some training case (for which ``case``
    is an integer) or testing case (for which
    ``case`` is a string), set up the physical
    problems and turbine locations defining the
    tidal farm modelling problem.

    For training data, these values are chosen
    randomly.
    """
    parameters.case = case
    parameters.discrete = discrete
    if isinstance(case, int):
        parameters.turbine_coords = []
        np.random.seed(100 * case)

        # Random depth from 20m to 100m
        parameters.depth = sample_uniform(20.0, 100.0)

        # Random inflow speed from 0.5 m/s to 6 m/s
        parameters.inflow_speed = sample_uniform(0.5, 6.0)

        # Random viscosity from 0.1 m^2/s to 1 m^2/s
        parameters.viscosity_coefficient = sample_uniform(0.1, 1.0)

        # Randomise turbine configuration such that all
        # turbines are at least 50m from the domain
        # boundaries and each other
        num_turbines = np.random.randint(1, 8)
        tc = parameters.turbine_coords
        i = 0
        while i < num_turbines:
            x = 50.0 + 1100.0 * np.random.rand()
            y = 50.0 + 400.0 * np.random.rand()
            valid = True
            for xyt in tc:
                if l2dist((x, y), xyt) < 50.0:
                    valid = False
            if valid:
                tc.append((x, y))
                i += 1
        return
    elif "aligned" in case:
        parameters.viscosity_coefficient = 0.5
        parameters.depth = 40.0
        parameters.inflow_speed = 5.0
        parameters.turbine_coords = [(456, 250), (744, 250)]
    elif "offset" in case:
        parameters.viscosity_coefficient = 0.5
        parameters.depth = 40.0
        parameters.inflow_speed = 5.0
        parameters.turbine_coords = [(456, 232), (744, 268)]
    elif "trench" in case:
        bmin, bmax = Constant(160.0), Constant(200.0)
        w = Constant(500.0)

        def bathy(mesh):
            y = SpatialCoordinate(mesh)[1] / w
            P0 = FunctionSpace(mesh, "DG", 0)
            b = Function(P0)
            b.interpolate(bmin + (bmax - bmin) * y * (1 - y))
            return b

        parameters.viscosity_coefficient = 2.0
        parameters.bathymetry = bathy
        parameters.inflow_speed = 10.0
        parameters.turbine_coords = [(456, 232), (744, 268)]
    elif "headland" in case:
        parameters.viscosity_coefficient = 100.0
        parameters.depth = 40.0
        parameters.inflow_speed = 5.0
        parameters.turbine_diameter = 80.0
        parameters.turbine_width = 100.0
        parameters.turbine_coords = [(600, 250)]
        parameters.correct_thrust = False
        parameters.solver_parameters = {
            "mat_type": "aij",
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1.0e-08,
            "snes_max_it": 100,
            "snes_monitor": None,
            "ksp_type": "preonly",
            "ksp_converged_reason": None,
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    elif "pipe" in case:
        u_in = Constant(5.0)
        parameters.inflow_speed = u_in
        w = Constant(200.0)

        def inflow(mesh):
            y = SpatialCoordinate(mesh)[1] / w
            yy = ((y - 0.5) / 0.5) ** 2
            u_expr = conditional(yy < 1, exp(1 - 1 / (1 - yy)), 0)
            return as_vector([u_expr, 0])

        parameters.viscosity_coefficient = 20.0
        parameters.depth = 40.0
        parameters.u_inflow = inflow
        parameters.ic = lambda mesh: as_vector([u_in, 0.0])
        parameters.turbine_coords = [(550, 300), (620, 390)]
        parameters.qoi_unit = "kW"
        parameters.density = Constant(1030.0 * 1.0e-03)
    else:
        raise ValueError(f"Test case {test_case} not recognised")

    if "reversed" in case:
        parameters.inflow_speed *= -1
