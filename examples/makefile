all: setup network snapshot convergence

MODELS			= turbine
TEST_CASES		= 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

.PHONY: plot_test_cases, plot_progress, plot_importance, plot_convergence

# --- Setup directories and meshes

setup: dir mesh plot_test_cases

dir:
	cd turbine && make dir

mesh:
	cd turbine && make mesh

plot_test_cases:
	for model in $(MODELS); do \
		python3 plot_test_cases.py $$model; \
	done

# --- Train the neural network

network: features train plot_progress plot_importance

features:
	for model in $(MODELS); do \
		for test_case in $(TEST_CASES); do \
			python3 run_adapt.py $$model $$test_case; \
			python3 run_adapt.py $$model $$test_case -anisotropic 1; \
		done; \
	done

train:
	for model in $(MODELS); do \
		python3 test_and_train.py $$model; \
	done

plot_progress:
	for model in $(MODELS); do \
		python3 plot_progress.py $$model; \
	done

plot_importance:
	for model in $(MODELS); do \
		python3 plot_importance.py $$model; \
	done

# --- Snapshots

snapshot:
	for model in $(MODELS); do \
		for test_case in $(TEST_CASES); do \
			python3 run_adapt.py $$model $$test_case; \
			python3 run_adapt.py $$model $$test_case -anisotropic 1; \
			python3 run_adapt_ml.py $$model $$test_case; \
			python3 run_adapt_ml.py $$model $$test_case -anisotropic 1; \
		done; \
	done

# --- Perform convergence analysis

convergence: uniform go ml plot_convergence

uniform:
	for model in $(MODELS); do \
		for test_case in $(TEST_CASES); do \
			python3 run_uniform_refinement.py $$model $$test_case; \
		done; \
	done

go:
	for model in $(MODELS); do \
		for test_case in $(TEST_CASES); do \
			python3 run_adaptation_loop.py $$model $$test_case; \
			python3 run_adaptation_loop.py $$model $$test_case -anisotropic 1; \
		done; \
	done

ml:
	for model in $(MODELS); do \
		for test_case in $(TEST_CASES); do \
			python3 run_adaptation_loop_ml.py $$model $$test_case; \
			python3 run_adaptation_loop_ml.py $$model $$test_case -anisotropic 1; \
		done; \
	done

plot_convergence:
	for model in $(MODELS); do \
		python3 plot_convergence.py $$model; \
	done

# --- Do profiling experiments

profile: profile_turbine

profile_turbine: profile_turbine_uni profile_turbine_go profile_turbine_ml

profile_turbine_uni:
	python3 run_fixed_mesh.py turbine 0 -optimise 1 -num_refinements 4 -log_view :logview.txt:ascii_flamegraph
	flamegraph.pl --title "Uniform refinement (test case 0)" logview.txt > turbine/outputs/uni0.svg && rm logview.txt

profile_turbine_go:
	python3 run_adapt.py turbine 0 -anisotropic 1 -optimise 1 -target_complexity 64000 -log_view :logview.txt:ascii_flamegraph
	flamegraph.pl --title "Goal-oriented adaptation (test case 0)" logview.txt > turbine/outputs/go0.svg && rm logview.txt

profile_turbine_ml:
	python3 run_adapt_ml.py turbine 0 -anisotropic 1 -optimise 1 -target_complexity 64000 -log_view :logview.txt:ascii_flamegraph
	flamegraph.pl --title "Data-driven adaptation (test case 0)" logview.txt > turbine/outputs/ml0.svg && rm logview.txt
