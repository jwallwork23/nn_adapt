L = 1200.0;
W = 500.0;
D = 18.0;
dx_outer = 20.0;
dx_inner = 20.0;

headland_x_scale = 0.2;
headland_y = 150;

site_x = 100;
site_y = 80;
site_x_start = L/2 - site_x/2;
site_x_end = site_x_start + site_x;

site_y_start = W/2 - site_y/2;
site_y_end = site_y_start + site_y;

Point(1) = {0, 0, 0, dx_outer};
Point(2) = {L, 0, 0, dx_outer};


// Headland
res = 100;
b = 10;
For k In {0:res:1}
    x = L/res*k;
    y = W - headland_y*Exp(-0.5*((headland_x_scale*(x-L/2))/b)^2);
	Point(10+k) = {x, y, 0, dx_outer};
EndFor
BSpline(100) = { 10 : res+10 };

// Domain boundary
Line(101) = {10, 1};
Line(102) = {1, 2};
Line(103) = {2, res+10};
Line Loop(104) = {100, -103, -102, -101};

// Generate site nodes
Point(111) = {site_x_start, site_y_start, 0, dx_inner};
Point(112) = {site_x_end, site_y_start, 0, dx_inner};
Point(113) = {site_x_end, site_y_end, 0, dx_inner};
Point(114) = {site_x_start, site_y_end, 0, dx_inner};
Line(105) = {111, 112};
Line(106) = {112, 113};
Line(107) = {113, 114};
Line(108) = {114, 111};
Line Loop(110) = {105, 106, 107, 108};
Plane Surface(111) = {104, 110};
Plane Surface(112) = {110};
Physical Line(1) = {101};  // inflow
Physical Line(2) = {103};  // outflow
Physical Line(3) = {100, 102};  // free-slip
Physical Line(4) = {};  // no-slip
Physical Line(5) = {};  // weakly reflective
Physical Surface(1) = {111};
Physical Surface(2) = {112};
