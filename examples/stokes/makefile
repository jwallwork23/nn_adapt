all: dir mesh

dir:
	mkdir -p outputs
	mkdir -p plots

mesh:
	cd meshes && make mesh

uniform:
	for i in 0 1 2 3 4; do python3 run_uniform_refinement.py $$i; done

go:
	for i in 0 1 2 3 4; do python3 run_adaptation_loop.py $$i; done

plot:
	for i in 0 1 2 3 4; do python3 plot_convergence.py $$i; done
