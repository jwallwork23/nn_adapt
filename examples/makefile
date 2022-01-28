all: setup network snapshot convergence

# --- Configurable parameters

MODELS				= turbine
NUM_TRAINING_CASES	= 16
TESTING_CASES		= aligned offset

# --- Parameters that should not need modifying

TRAINING_CASES	= $(shell seq 1 ${NUM_TRAINING_CASES})
CASES			= ${TRAINING_CASES} ${TESTING_CASES}

.PHONY: plot_test_cases, plot_progress, plot_importance, plot_convergence

# --- Setup directories and meshes

setup: dir mesh plot_test_cases

dir:
	for model in $(MODELS); do \
		mkdir -p $$model; \
		cd $$model && touch __init__.py && mkdir -p data outputs plots && cd ..; \
	done

mesh:
	for model in $(MODELS); do \
		for test_case in $(CASES); do \
			python3 meshgen.py $$model $$test_case; \
			cd $$model/meshes && gmsh -2 $$test_case.geo && cd ../..; \
		done; \
	done

plot_test_cases:
	for model in $(MODELS); do \
		python3 plot_test_cases.py $$model; \
	done

clean:
	for model in $(MODELS); do \
		cd $$model && rm -rf data outputs plots __pycache__ && cd ..; \
	done

# --- Train the neural network

network: features train plot_progress plot_importance

features:
	for model in $(MODELS); do \
		for test_case in $(TRAINING_CASES); do \
			python3 run_adapt.py $$model $$test_case; \
			python3 run_adapt.py $$model $$test_case -anisotropic 1; \
		done; \
	done

train:
	for model in $(MODELS); do \
		python3 test_and_train.py $$model $(NUM_TRAINING_CASES); \
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
		for test_case in $(CASES); do \
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
		for test_case in $(TESTING_CASES); do \
			python3 run_uniform_refinement.py $$model $$test_case; \
		done; \
	done

go:
	for model in $(MODELS); do \
		for test_case in $(TESTING_CASES); do \
			python3 run_adaptation_loop.py $$model $$test_case; \
			python3 run_adaptation_loop.py $$model $$test_case -anisotropic 1; \
		done; \
	done

ml:
	for model in $(MODELS); do \
		for test_case in $(TESTING_CASES); do \
			python3 run_adaptation_loop_ml.py $$model $$test_case; \
			python3 run_adaptation_loop_ml.py $$model $$test_case -anisotropic 1; \
		done; \
	done

plot_convergence:
	for model in $(MODELS); do \
		for test_case in $(TESTING_CASES); do \
			python3 plot_convergence.py $$model $$test_case; \
			done; \
	done

# --- Do profiling experiments  # TODO: generalise

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
