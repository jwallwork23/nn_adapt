// Domain and turbine specification
L = 1200.0;
W = 500.0;
D = 18.0;
dx_outer = 40.0;
dx_inner = 8.0;

// Domain and turbine footprints
Point(1) = {0, 0, 0, dx_outer};
Point(2) = {L, 0, 0, dx_outer};
Point(3) = {L, W, 0, dx_outer};
Point(4) = {0, W, 0, dx_outer};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Physical Line(1) = {4};   // Left boundary
Physical Line(2) = {2};   // Right boundary
Physical Line(3) = {1,3}; // Sides
Line Loop(1) = {1, 2, 3, 4};  // outside loop
Point(5) = {50-D/2, W/2-D/2, 0., dx_inner};
Point(6) = {50+D/2, W/2-D/2, 0., dx_inner};
Point(7) = {50+D/2, W/2+D/2, 0., dx_inner};
Point(8) = {50-D/2, W/2+D/2, 0., dx_inner};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Line Loop(2) = {5, 6, 7, 8};  // inside loop 1

// Surfaces
Plane Surface(1) = {1, 2};
Plane Surface(2) = {2};
Physical Surface(1) = {1};  // outside turbine
Physical Surface(2) = {2};  // inside turbine 1
