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
	for i in 0 1 2 3 4; do python3 run_uniform_refinement.py stokes $$i; done

go:
	for i in 0 1 2 3 4; do python3 run_adaptation_loop.py stokes $$i; done

ml:
	for i in 0 1 2 3 4; do python3 run_adaptation_loop_ml.py stokes $$i; done

plot:
	for i in 0 1 2 3 4; do python3 plot_convergence.py stokes $$i; done

profile:
	python3 run_adapt_ml.py stokes 0 -log_view :logview.txt:ascii_flamegraph
	flamegraph.pl --title "Data-driven Stokes 0" logview.txt > stokes/outputs/flamegraph.svg && rm logview.txt
