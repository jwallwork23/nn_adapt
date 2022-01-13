// Domain and turbine specification
L = 1200.0;
W = 500.0;
D = 18.0;
dx_outer = 40.0;
dx_inner = 8.0;
xt0=456.0;  // x-location of turbine 1
yt0=250.0;  // y-location of turbine 1
xt1=744.0;  // x-location of turbine 2
yt1=250.0;  // y-location of turbine 2

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
Point(5) = {xt0-D/2, yt0-3*D/2, 0., dx_inner};
Point(6) = {xt0+D/2, yt0-3*D/2, 0., dx_inner};
Point(7) = {xt0+D/2, yt0-D/2, 0., dx_inner};
Point(8) = {xt0-D/2, yt0-D/2, 0., dx_inner};
Point(9) = {xt1-D/2, yt1+D/2, 0., dx_inner};
Point(10) = {xt1+D/2, yt1+D/2, 0., dx_inner};
Point(11) = {xt1+D/2, yt1+3*D/2, 0., dx_inner};
Point(12) = {xt1-D/2, yt1+3*D/2, 0., dx_inner};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Line(9) = {9, 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 9};
Line Loop(2) = {5, 6, 7, 8};  // inside loop 1
Line Loop(3) = {9, 10, 11, 12};  // inside loop 2

// Surfaces
Plane Surface(1) = {1, 2, 3};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Physical Surface(1) = {1};  // outside turbine
Physical Surface(2) = {2};  // inside turbine 1
Physical Surface(3) = {3};  // inside turbine 2
