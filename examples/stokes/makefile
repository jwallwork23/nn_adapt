all: dir mesh

dir:
	mkdir -p data
	mkdir -p outputs
	mkdir -p plots

mesh:
	cd meshes && make mesh

go:
	for i in 0 1 2 3 4; do python3 run_adaptation_loop.py $$i; done
