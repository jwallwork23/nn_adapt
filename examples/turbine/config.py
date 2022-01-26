from models.turbine import *


def initialise(i):
    if i < 14:
        np.random.seed(3*i)

        # Random depth from 20m to 100m
        parameters.depth = 20.0 + 80.0*np.random.rand()

        # Random viscosity from 0.001 to 10
        significand = 1.0 + np.random.rand()
        exponent = np.random.randint(-3, 1)
        parameters.viscosity.assign(significand*10**exponent)

    elif i in [14, 15]:
        parameters.viscosity.assign(0.5)
        parameters.depth = 40.0

    # Set coordinates
    parameters.turbine_coords = [

        # Training and validation
        [(600, 250)],
        [(600, 375)],
        [(50, 250)],
        [(1100, 250)],
        [(206, 300), (796, 300)],
        [(456, 220), (456, 280)],
        [(256, 250), (544, 250)],
        [(456, 44), (456, 456)],
        [(400, 268), (600, 250), (800, 232)],
        [(420, 250), (600, 232), (780, 250)],
        [(650, 400), (700, 400), (750, 400)],
        [(100, 450), (600, 250), (1100, 50)],
        [(500, 100), (500, 200), (500, 300), (500, 400)],
        [(500, 286), (600, 250), (700, 214), (500, 214), (700, 286)],

        # Testing
        [(456, 250), (744, 250)],  # aligned
        [(456, 232), (744, 268)],  # offset
    ][i]
