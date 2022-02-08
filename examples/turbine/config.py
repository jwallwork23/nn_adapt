from models.turbine import *


testing_cases = ["aligned", "offset"]


def l2dist(xy, xyt):
    r"""
    Usual :math:`\ell_2` distance between
    two points in Euclidean space.
    """
    diff = np.array(xy) - np.array(xyt)
    return np.sqrt(np.dot(diff, diff))


def initialise(case):
    """
    Given some training case (for which ``case``
    is an integer) or testing case (for which
    ``case`` is a string), set up the physical
    problems and turbine locations defining the
    tidal farm modelling problem.

    For training data, these values are chosen
    randomly.
    """
    if case == "aligned":
        parameters.viscosity.assign(0.5)
        parameters.depth = 40.0
        parameters.inflow_speed = 5.0
        parameters.turbine_coords = [(456, 250), (744, 250)]
        return
    elif case == "offset":
        parameters.viscosity.assign(0.5)
        parameters.depth = 40.0
        parameters.inflow_speed = 5.0
        parameters.turbine_coords = [(456, 232), (744, 268)]
        return
    else:
        assert isinstance(case, int)
        parameters.turbine_coords = []
    np.random.seed(100 * case)

    # Random depth from 20m to 100m
    parameters.depth = 20.0 + 80.0 * np.random.rand()

    # Random inflow speed from 0.5 m/s to 5 m/s
    parameters.inflow_speed = 0.5 + 4.5 * np.random.rand()

    # Random viscosity from 0.001 to 10
    significand = 1.0 + np.random.rand()
    exponent = np.random.randint(-3, 1)
    parameters.viscosity.assign(significand * 10**exponent)

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
