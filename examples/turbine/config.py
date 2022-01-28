from models.turbine import *


def l2dist(xy, xyt):
    diff = np.array(xy) - np.array(xyt)
    return np.sqrt(np.dot(diff, diff))


def initialise(test_case):
    if test_case == 'aligned':
        parameters.viscosity.assign(0.5)
        parameters.depth = 40.0
        parameters.inflow_speed = 5.0
        parameters.turbine_coords = [(456, 250), (744, 250)]  # aligned
        return
    elif test_case == 'offset':
        parameters.viscosity.assign(0.5)
        parameters.depth = 40.0
        parameters.inflow_speed = 5.0
        parameters.turbine_coords = [(456, 232), (744, 268)]  # offset
        return
    else:
        assert isinstance(test_case, int)
        parameters.turbine_coords = []
    np.random.seed(100*test_case)

    # Random depth from 20m to 100m
    parameters.depth = 20.0 + 80.0*np.random.rand()

    # Random inflow speed from 0.5 m/s to 5 m/s
    parameters.inflow_speed = 0.5 + 4.5*np.random.rand()

    # Random viscosity from 0.001 to 10
    significand = 1.0 + np.random.rand()
    exponent = np.random.randint(-3, 1)
    parameters.viscosity.assign(significand*10**exponent)

    # Randomise turbine configuration
    num_turbines = np.random.randint(1, 8)
    tc = parameters.turbine_coords
    i = 0
    while i < num_turbines:
        x = 50.0 + 1100.0*np.random.rand()
        y = 50.0 + 400.0*np.random.rand()
        valid = True
        for xyt in tc:
            if l2dist((x, y), xyt) < 50.0:
                valid = False
        if valid:
            tc.append((x, y))
            i += 1
