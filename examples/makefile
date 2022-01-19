all: dir mesh

.PHONY: plot_test_cases, plot_weights, plot_progress, plot_convergence

dir:
	cd turbine && make dir

mesh:
	cd turbine && make mesh

plot_test_cases:
	for model in turbine; do \
		python3 plot_test_cases.py $$model; \
	done

# --- Train the neural network

setup: features train plot_weights plot_progress

features:
	for model in turbine; do \
		for i in `seq 0 11`; do \
			python3 run_adapt.py $$model $$i; \
			python3 run_adapt.py $$model $$i -anisotropic 1; \
		done; \
	done

train:
	for model in turbine; do \
		python3 test_and_train.py $$model; \
	done

plot_weights:
	for model in turbine; do \
		python3 plot_weights.py $$model; \
	done

plot_progress:
	for model in turbine; do \
		python3 plot_progress.py $$model; \
	done

# --- Snapshots

snapshot:
	for model in turbine; do \
		for i in `seq 0 11`; do \
			python3 run_adapt_ml.py $$model $$i; \
			python3 run_adapt_ml.py $$model $$i -anisotropic 1; \
		done; \
	done

# --- Perform convergence analysis

convergence: uniform go ml plot_convergence

uniform:
	for model in turbine; do \
		for i in `seq 0 11`; do \
			python3 run_uniform_refinement.py $$model $$i; \
		done; \
	done

go:
	for model in turbine; do \
		for i in `seq 0 11`; do \
			python3 run_adaptation_loop.py $$model $$i; \
			python3 run_adaptation_loop.py $$model $$i -anisotropic 1; \
		done; \
	done

ml:
	for model in turbine; do \
		for i in `seq 0 11`; do \
			python3 run_adaptation_loop_ml.py $$model $$i; \
			python3 run_adaptation_loop_ml.py $$model $$i -anisotropic 1; \
		done; \
	done

plot_convergence:
	for model in turbine; do \
		python3 plot_convergence.py $$model; \
	done

# --- Do profiling experiments

profile: profile_turbine

profile_turbine: profile_turbine_go profile_turbine_ml

profile_turbine_go:
	python3 run_adapt.py turbine 0 -target_complexity 32000 -log_view :logview.txt:ascii_flamegraph
	flamegraph.pl --title "Goal-Oriented (test case 0)" logview.txt > turbine/outputs/go0.svg && rm logview.txt

profile_turbine_ml:
	python3 run_adapt_ml.py turbine 0 -target_complexity 32000 -log_view :logview.txt:ascii_flamegraph
	flamegraph.pl --title "Data-driven (test case 0)" logview.txt > turbine/outputs/ml0.svg && rm logview.txt
