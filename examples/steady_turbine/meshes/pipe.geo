D = 18.0;
dx_outer = 20.0;
dx_inner = 20.0;
xt0 = 550.0;  // x-location of turbine 0
yt0 = 300.0;  // y-location of turbine 0
xt1 = 620.0;  // x-location of turbine 1
yt1 = 390.0;  // y-location of turbine 1
W = 700;  // Width of domain
w = 200;  // Width of channel

sqrt2 = 1.414213562373095;

// Lower points
Point(1) = {0,    0,   0, dx_outer};
Point(2) = {200,  0,   0, dx_outer}; // First spline point
Point(3) = {400,  0,   0, dx_outer};
Point(4) = {600,  W/2-0.75*w, 0, dx_outer};
Point(5) = {800,  W-w, 0, dx_outer};
Point(6) = {1000, W-w, 0, dx_outer}; // Last spline point
Point(7) = {1200, W-w, 0, dx_outer};
// Upper points
Point( 8) = {1200, W, 0, dx_outer};
Point( 9) = {1000, W, 0, dx_outer}; // First spline point
Point(10) = {800,  W, 0, dx_outer};
Point(11) = {600,  W/2+0.75*w, 0, dx_outer};
Point(12) = {400,  w, 0, dx_outer};
Point(13) = {200,  w, 0, dx_outer}; // Last spline point
Point(14) = {0,    w, 0, dx_outer};

// Edges
Line(1) = {1, 2};
BSpline(2) = {2, 3, 4, 5, 6};
Line(3) = {6, 7};
Line(4) = {7, 8};
Line(5) = {8, 9};
BSpline(6) = {9, 10, 11, 12, 13};
Line(7) = {13, 14};
Line(8) = {14,  1};

// Boundary and physical curves
Curve Loop(9) = {1, 2, 3, 4, 5, 6, 7, 8};
Physical Curve("Inflow", 1) = {8};
Physical Curve("Outflow", 2) = {4};
Physical Curve("Sides", 4) = {1, 2, 3, 5, 6, 7};

Point(15) = {xt0-D/sqrt2, yt0, 0., dx_inner};
Point(16) = {xt0, yt0-D/sqrt2, 0., dx_inner};
Point(17) = {xt0+D/sqrt2, yt0, 0., dx_inner};
Point(18) = {xt0, yt0+D/sqrt2, 0., dx_inner};
Line(15) = {15, 16};
Line(16) = {16, 17};
Line(17) = {17, 18};
Line(18) = {18, 15};
Line Loop(10) = {15, 16, 17, 18};

Point(19) = {xt1-D/sqrt2, yt1, 0., dx_inner};
Point(20) = {xt1, yt1-D/sqrt2, 0., dx_inner};
Point(21) = {xt1+D/sqrt2, yt1, 0., dx_inner};
Point(22) = {xt1, yt1+D/sqrt2, 0., dx_inner};
Line(19) = {19, 20};
Line(20) = {20, 21};
Line(21) = {21, 22};
Line(22) = {22, 19};
Line Loop(11) = {19, 20, 21, 22};

// Domain and physical surface
Plane Surface(1) = {9, 10, 11};
Physical Surface("Pipe", 1) = {1};
Plane Surface(2) = {10};
Physical Surface(2) = {2};  // inside turbine 1
Plane Surface(3) = {11};
Physical Surface(3) = {3};  // inside turbine 2
