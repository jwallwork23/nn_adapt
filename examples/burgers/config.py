from models.burgers import *
import numpy as np


testing_cases = ["demo"]


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
    problems defining the Burgers problem.

    For training data, these values are chosen
    randomly.
    """
    parameters.case = case
    parameters.discrete = discrete
    if isinstance(case, int):
        parameters.turbine_coords = []
        np.random.seed(100 * case)

        # Random initial speed from 0.01 m/s to 6 m/s
        parameters.initial_speed = sample_uniform(0.01, 6.0)

        # Random viscosity from 0.00001 m^2/s to 1 m^2/s
        parameters.viscosity_coefficient = sample_uniform(0.1, 1.0) * 10 ** np.random.randint(-3, 1)
        return
    elif "demo" in case:
        parameters.viscosity_coefficient = 0.0001
        parameters.initial_speed = 1.0
    else:
        raise ValueError(f"Test case {test_case} not recognised")

    if "reversed" in case:
        parameters.initial_speed *= -1
