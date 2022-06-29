all: setup network convergence profile snapshot

# --- Configurable parameters

APPROACHES			= anisotropic
MODEL				= turbine
NUM_TRAINING_CASES	= 100
TESTING_CASES		= $(shell cat $(MODEL)/testing_cases.txt)
PETSC_OPTIONS		= -dm_plex_metric_hausdorff_number 1

# --- Parameters that should not need modifying

TRAINING_CASES	= $(shell seq 1 ${NUM_TRAINING_CASES})
CASES			= ${TRAINING_CASES} ${TESTING_CASES}

# --- Setup directories and meshes

setup: dir mesh plot_config

dir:
	mkdir -p $(MODEL)/data
	mkdir -p $(MODEL)/outputs
	mkdir -p $(MODEL)/plots
	touch $(MODEL)/__init__.py
	for test_case in $(TESTING_CASES); do \
		mkdir -p $(MODEL)/outputs/$$test_case; \
	done

mesh:
	touch timing.log
	d=$$(date +%s) && \
	for test_case in $(CASES); do \
		python3 meshgen.py $(MODEL) $$test_case; \
		gmsh -2 -algo pack $(MODEL)/meshes/$$test_case.geo -o $(MODEL)/meshes/$$test_case.msh; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Meshes built in $$(($$(date +%s)-d)) seconds" >> timing.log

plot_config:
	python3 plot_config.py $(MODEL) 'train'
	python3 plot_config.py $(MODEL) 'test'

clean:
	rm -rf $(MODEL)/data
	rm -rf $(MODEL)/outputs
	rm -rf $(MODEL)/plots
	rm -rf $(MODEL)/__pycache__

# --- Train the neural network

network: features train plot_progress plot_importance

features:
	touch timing.log
	d=$$(date +%s) && \
	for test_case in $(TRAINING_CASES); do \
		for approach in $(APPROACHES); do \
			python3 run_adapt.py $(MODEL) $$test_case -a $$approach --no_outputs $(PETSC_OPTIONS); \
		done; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Features generated in $$(($$(date +%s)-d)) seconds" >> timing.log
	echo "" >> timing.log

train:
	touch timing.log
	d=$$(date +%s) && \
	python3 test_and_train.py -m $(MODEL) -n $(NUM_TRAINING_CASES) --tag all && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Training completed in $$(($$(date +%s)-d)) seconds" >> timing.log && \
	echo "" >> timing.log

plot_progress:
	python3 plot_progress.py $(MODEL) --tag all

plot_importance:
	python3 compute_importance.py $(MODEL) $(NUM_TRAINING_CASES) --tag all
	python3 plot_importance.py $(MODEL) $(NUM_TRAINING_CASES) --tag all

# --- Snapshots

snapshot:
	touch timing.log
	d=$$(date +%s) && \
	for test_case in $(TESTING_CASES); do \
		for approach in $(APPROACHES); do \
			python3 run_adapt.py $(MODEL) $$test_case -a $$approach $(PETSC_OPTIONS); \
			python3 run_adapt_ml.py $(MODEL) $$test_case -a $$approach --tag all $(PETSC_OPTIONS); \
		done; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Snapshots generated in $$(($$(date +%s)-d)) seconds" >> timing.log
	echo "" >> timing.log

# --- Perform convergence analysis

convergence: uniform go ml plot_convergence

uniform:
	touch timing.log
	d=$$(date +%s) && \
	for test_case in $(TESTING_CASES); do \
		python3 run_uniform_refinement.py $(MODEL) $$test_case; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Uniform refinement completed in $$(($$(date +%s)-d)) seconds" >> timing.log
	echo "" >> timing.log

go:
	touch timing.log
	d=$$(date +%s) && \
	for test_case in $(TESTING_CASES); do \
		for approach in $(APPROACHES); do \
			python3 run_adaptation_loop.py $(MODEL) $$test_case -a $$approach $(PETSC_OPTIONS); \
		done; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Goal-oriented adaptation completed in $$(($$(date +%s)-d)) seconds" >> timing.log
	echo "" >> timing.log

ml:
	touch timing.log
	d=$$(date +%s) && \
	for test_case in $(TESTING_CASES); do \
		for approach in $(APPROACHES); do \
			python3 run_adaptation_loop_ml.py $(MODEL) $$test_case -a $$approach --tag all $(PETSC_OPTIONS); \
		done; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Data-driven adaptation completed in $$(($$(date +%s)-d)) seconds" >> timing.log
	echo "" >> timing.log

plot_convergence:
	for test_case in $(TESTING_CASES); do \
		python3 plot_convergence.py $(MODEL) $$test_case; \
	done

# --- Do profiling experiments

profile:
	for i in 1 2 3 4 5; do \
		make profile_uni; \
	done && \
	for i in 1 2 3 4 5; do \
		make profile_go; \
	done && \
	for i in 1 2 3 4 5; do \
		make profile_ml; \
	done

profile_uni:
	touch timing.log
	d=$$(date +%s) && \
	for test_case in $(TESTING_CASES); do \
		python3 run_fixed_mesh.py $(MODEL) $$test_case --optimise --num_refinements 4 $(PETSC_OPTIONS) -log_view :logview.txt:ascii_flamegraph; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Uniform refinement profiling run completed in $$(($$(date +%s)-d)) seconds" >> timing.log && \
	echo "" >> timing.log
	for test_case in $(TESTING_CASES); do \
		flamegraph.pl --title "Uniform refinement ($$test_case)" logview.txt > $(MODEL)/outputs/$$test_case/uni.svg && \
		rm logview.txt; \
	done

profile_go:
	touch timing.log
	d=$$(date +%s) && \
	for test_case in $(TESTING_CASES); do \
		python3 run_adapt.py $(MODEL) $$test_case -a anisotropic --optimise --target_complexity 64000 $(PETSC_OPTIONS) -log_view :logview.txt:ascii_flamegraph; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Goal-oriented adaptation profiling run completed in $$(($$(date +%s)-d)) seconds" >> timing.log
	echo "" >> timing.log
	for test_case in $(TESTING_CASES); do \
		flamegraph.pl --title "Goal-oriented adaptation ($$test_case)" logview.txt > $(MODEL)/outputs/$$test_case/go.svg && \
		rm logview.txt; \
	done

profile_ml:
	touch timing.log
	d=$$(date +%s) && \
	for test_case in $(TESTING_CASES); do \
		python3 run_adapt_ml.py $(MODEL) $$test_case -a anisotropic --optimise --target_complexity 64000 $(PETSC_OPTIONS) --tag all -log_view :logview.txt:ascii_flamegraph; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Data-driven adaptation profiling run completed in $$(($$(date +%s)-d)) seconds" >> timing.log
	echo "" >> timing.log
	for test_case in $(TESTING_CASES); do \
		flamegraph.pl --title "Data-driven adaptation ($$test_case)" logview.txt > $(MODEL)/outputs/$$test_case/ml.svg && \
		rm logview.txt; \
	done
