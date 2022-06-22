def generate_geo(config, reverse=False):
    """
    Given a configuration object for a given training
    or testing case, generate a gmsh geometry file
    defining the initial mesh, with all turbines
    meshed explicitly.

    :arg config: the configuration file
    :kwarg reverse: should the flow direction be reversed?
    """
    if config.parameters.case == "pipe":
        return
    tc = config.parameters.turbine_coords
    num_turbines = len(tc)
    f = """// Domain and turbine specification
L = 1200.0;
W = 500.0;
D = 18.0;
dx_outer = 20.0;
dx_inner = 20.0;
"""
    for i, xy in enumerate(tc):
        f += "xt%d = %f;  // x-location of turbine %d\n" % (i, xy[0], i)
        f += "yt%d = %f;  // y-location of turbine %d\n" % (i, xy[1], i)
    f += """
// Domain and turbine footprints
Point(1) = {0, 0, 0, dx_outer};
Point(2) = {L, 0, 0, dx_outer};
Point(3) = {L, W, 0, dx_outer};
Point(4) = {0, W, 0, dx_outer};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Physical Line(1) = {%d};   // Left boundary
Physical Line(2) = {%d};   // Right boundary
Physical Line(3) = {1, 3}; // Sides
Line Loop(1) = {1, 2, 3, 4};  // outside loop
""" % (
        2 if reverse else 4,
        4 if reverse else 2,
    )
    i = 5
    j = 2
    for k in range(num_turbines):
        f += """
Point(%d) = {xt%d-D/2, yt%d-D/2, 0., dx_inner};
Point(%d) = {xt%d+D/2, yt%d-D/2, 0., dx_inner};
Point(%d) = {xt%d+D/2, yt%d+D/2, 0., dx_inner};
Point(%d) = {xt%d-D/2, yt%d+D/2, 0., dx_inner};
Line(%d) = {%d, %d};
Line(%d) = {%d, %d};
Line(%d) = {%d, %d};
Line(%d) = {%d, %d};
Line Loop(%d) = {%d, %d, %d, %d};
""" % (
            i,
            k,
            k,
            i + 1,
            k,
            k,
            i + 2,
            k,
            k,
            i + 3,
            k,
            k,
            i,
            i,
            i + 1,
            i + 1,
            i + 1,
            i + 2,
            i + 2,
            i + 2,
            i + 3,
            i + 3,
            i + 3,
            i,
            j,
            i,
            i + 1,
            i + 2,
            i + 3,
        )
        i += 4
        j += 1
    f += """
// Surfaces
Plane Surface(1) = %s;
Physical Surface(1) = {1};  // outside turbines
""" % set(
        range(1, num_turbines + 2)
    )
    for i in range(1, num_turbines + 1):
        f += "Plane Surface(%d) = {%d};\n" % (i + 1, i + 1)
        f += "Physical Surface(%d) = {%d};  // inside turbine %d\n" % (i + 1, i + 1, i)
    return f[:-1]
