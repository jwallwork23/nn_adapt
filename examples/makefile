all: dir mesh

dir:
	cd stokes && make dir

mesh:
	cd stokes && make mesh

features:
	for i in 0 1 2 3 4; do python3 run_adapt.py stokes $$i; done

train:
	python3 test_and_train.py stokes

uniform:
	for i in 0 1 2 3 4; do python3 run_uniform_refinement.py $$i; done

plot:
	for i in 0 1 2 3 4; do python3 plot_convergence.py stokes $$i; done
