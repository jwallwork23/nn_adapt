all: dir mesh

.PHONY: plot_progress, plot_convergence

dir:
	cd stokes && make dir

mesh:
	cd stokes && make mesh

# --- Train the neural network

setup: features train plot_progress

features:
	for model in stokes; do \
		for i in 0 1 2 3 4; do \
			python3 run_adapt.py $$model $$i; \
		done; \
	done

train:
	for model in stokes; do \
		python3 test_and_train.py $$model; \
	done

plot_progress:
	for model in stokes; do \
		python3 plot_progress.py $$model; \
	done

# --- Perform convergence analysis

convergence: uniform go ml plot_convergence

uniform:
	for model in stokes; do \
		for i in 0 1 2 3 4; do \
			python3 run_uniform_refinement.py $$model $$i; \
		done; \
	done

go:
	for model in stokes; do \
		for i in 0 1 2 3 4; do \
			python3 run_adaptation_loop.py $$model $$i; \
		done; \
	done

ml:
	for model in stokes; do \
		for i in 0 1 2 3 4; do \
			python3 run_adaptation_loop_ml.py $$model $$i; \
		done; \
	done

plot_convergence:
	for model in stokes; do \
		for i in 0 1 2 3 4; \
			do python3 plot_convergence.py $$model $$i; \
		done; \
	done

# --- Do profiling experiments

profile:
	python3 run_adapt.py stokes 0 -log_view :logview.txt:ascii_flamegraph
	flamegraph.pl --title "Goal-Oriented Stokes 0" logview.txt > stokes/outputs/go0.svg && rm logview.txt
	python3 run_adapt_ml.py stokes 0 -log_view :logview.txt:ascii_flamegraph
	flamegraph.pl --title "Data-driven Stokes 0" logview.txt > stokes/outputs/ml0.svg && rm logview.txt
