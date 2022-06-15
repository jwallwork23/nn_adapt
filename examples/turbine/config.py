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
    parameters.discrete = discrete
    if "aligned" in case:
        parameters.viscosity_coefficient = 0.5
        parameters.depth = 40.0
        parameters.inflow_speed = 5.0
        parameters.turbine_coords = [(456, 250), (744, 250)]
    elif "offset" in case:
        parameters.viscosity_coefficient = 0.5
        parameters.depth = 40.0
        parameters.inflow_speed = 5.0
        parameters.turbine_coords = [(456, 232), (744, 268)]
    else:
        assert isinstance(case, int)
        parameters.turbine_coords = []
        np.random.seed(100 * case)

        # Random depth from 20m to 100m
        parameters.depth = sample_uniform(20.0, 10.0)
        # TODO: spatially varying

        # Random inflow speed from 0.5 m/s to 6 m/s
        parameters.inflow_speed = sample_uniform(0.5, 6.0)

        # Random viscosity from 0.01 m^2/s to 10 m^2/s
        parameters.viscosity_coefficient = sample_uniform(0.1, 1.0)
        # TODO: spatially varying

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

    if "reversed" in case:
        parameters.inflow_speed *= -1
