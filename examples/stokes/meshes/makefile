all: mesh gradate

mesh:
	# for i in 0 1 2 3 4; do for j in 0 1 2 3 4 5; do gmsh -2 $$i.$$j.geo; done; done
	for i in 0 1 2 3 4; do for j in 5; do gmsh -2 $$i.$$j.geo; done; done

gradate:
	for i in 0 1 2 3 4; do python3 postproc.py $$i -dm_plex_prescribed_boundary_labels 4 -dm_plex_prescribed_boundary_sizes 0.1; done
