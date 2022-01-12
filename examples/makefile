all: dir mesh

.PHONY: plot_progress, plot_convergence

dir:
	cd stokes && make dir
	cd turbine && make dir

mesh:
	cd stokes && make mesh
	cd turbine && make mesh

# --- Train the neural network

setup: features train plot_progress

features:
	for model in stokes turbine; do \
		for i in `seq 0 9`; do \
			python3 run_adapt.py $$model $$i; \
			python3 run_adapt.py $$model $$i -anisotropic 1; \
		done; \
	done

train:
	for model in stokes turbine; do \
		python3 test_and_train.py $$model; \
	done

plot_progress:
	for model in stokes turbine; do \
		python3 plot_progress.py $$model; \
	done

# --- Snapshots

snapshot:
	for model in stokes turbine; do \
		for i in `seq 0 9`; do \
			python3 run_adapt_ml.py $$model $$i; \
		done; \
	done

# --- Perform convergence analysis

convergence: uniform go ml plot_convergence

uniform:
	for model in stokes turbine; do \
		for i in `seq 0 9`; do \
			python3 run_uniform_refinement.py $$model $$i; \
		done; \
	done

go:
	for model in stokes turbine; do \
		for i in `seq 0 9`; do \
			python3 run_adaptation_loop.py $$model $$i; \
		done; \
	done

ml:
	for model in stokes turbine; do \
		for i in `seq 0 9`; do \
			python3 run_adaptation_loop_ml.py $$model $$i; \
		done; \
	done

plot_convergence:
	for model in stokes turbine; do \
		for i in `seq 0 9`; do \
			do python3 plot_convergence.py $$model $$i; \
		done; \
	done

# --- Do profiling experiments

profile: profile_stokes profile_turbine

profile_stokes: profile_stokes_go profile_stokes_ml

profile_stokes_go:
	python3 run_adapt.py stokes 0 -log_view :logview.txt:ascii_flamegraph
	flamegraph.pl --title "Goal-Oriented Stokes 0" logview.txt > stokes/outputs/go0.svg && rm logview.txt

profile_stokes_ml:
	python3 run_adapt_ml.py stokes 0 -log_view :logview.txt:ascii_flamegraph
	flamegraph.pl --title "Data-driven Stokes 0" logview.txt > stokes/outputs/ml0.svg && rm logview.txt

profile_turbine: profile_turbine_go profile_turbine_ml

profile_turbine_go:
	python3 run_adapt.py turbine 0 -log_view :logview.txt:ascii_flamegraph
	flamegraph.pl --title "Goal-Oriented 2-turbine 0" logview.txt > turbine/outputs/go0.svg && rm logview.txt

profile_turbine_ml:
	python3 run_adapt_ml.py turbine 0 -log_view :logview.txt:ascii_flamegraph
	flamegraph.pl --title "Data-driven 2-turbine 0" logview.txt > turbine/outputs/ml0.svg && rm logview.txt
