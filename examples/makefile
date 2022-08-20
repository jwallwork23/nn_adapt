all: setup network test

# --- Configurable parameters

APPROACHES			= anisotropic
MODEL				= pyroteus_burgers
NUM_TRAINING_CASES	= 1
TESTING_CASES		= $(shell cat $(MODEL)/testing_cases.txt)
PETSC_OPTIONS		= -dm_plex_metric_hausdorff_number 1
TAG					= all

# --- Parameters that should not need modifying

TRAINING_CASES	= $(shell seq 1 ${NUM_TRAINING_CASES})
CASES			= ${TRAINING_CASES} ${TESTING_CASES}

# --- Setup directories and meshes

setup: dir mesh plot_config

# Create the directory structure
# ==============================
#
#   $(MODEL)
#      ├── data
#      ├── outputs
#      │     └── $(TESTING_CASES)
#      └── plots
dir:
	mkdir -p $(MODEL)/data
	mkdir -p $(MODEL)/outputs
	mkdir -p $(MODEL)/plots
	for case in $(TESTING_CASES); do \
		mkdir -p $(MODEL)/outputs/$$case; \
	done

# Generate meshes
# ===============
#
# Meshes are generated for all training and testing cases.
#  * First, a gmsh geometry file is generated using the
#    `meshgen.py` script. The definitions of these cases
#    are based on the contents of $(MODEL)/config.py.
#    For the `turbine` case, the training data is generated
#    randomly.
#  * Then the geometry files are used to construct meshes
#    in the .msh format.
#
# Gmsh is set to use the "pack" algorithm, which means that
# the initial meshes are quasi-uniform. That is, they are as
# close to uniform as they can be, given that the turbines
# are to be explicitly meshed.
mesh:
	touch timing.log
	d=$$(date +%s) && \
	for case in $(CASES); do \
		python3 meshgen.py $(MODEL) $$case; \
		if [ -e $(MODEL)/meshes/$$case.geo ]; then \
			gmsh -2 -algo pack $(MODEL)/meshes/$$case.geo -o $(MODEL)/meshes/$$case.msh; \
	    fi; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Meshes built in $$(($$(date +%s)-d)) seconds" >> timing.log

# Plot configurations
# ===================
#
# Plot the configurations for a subset of the training cases
# and the testing cases that are listed in $(MODEL)/config.py.
# The domain geometry and turbine locations are shown, along
# with the physical parameters used.
plot_config:
	python3 plot_config.py $(MODEL) 'train'
	python3 plot_config.py $(MODEL) 'test'

# Clean the model directory
# =========================
#
# Delete all logs, data, outputs, plots and compiled code associated
# with the model. Note that this is a very destructive thing to do!
clean:
	rm -rf timing.log
	rm -rf $(MODEL)/data
	rm -rf $(MODEL)/outputs
	rm -rf $(MODEL)/plots
	rm -rf $(MODEL)/__pycache__

# --- Construct the neural network

network: features train plot_progress plot_importance

# Generate feature data
# =====================
#
# This involves applying mesh adaptation to all of the cases in the
# training data. In each case, feature data and "target" error indicator
# data are extracted and saved to file.
features:
	touch timing.log
	d=$$(date +%s) && \
	for case in $(TRAINING_CASES); do \
		for approach in $(APPROACHES); do \
			python3 run_adapt.py $(MODEL) $$case -a $$approach --no_outputs $(PETSC_OPTIONS); \
		done; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Features generated in $$(($$(date +%s)-d)) seconds" >> timing.log
	echo "" >> timing.log

# Train the network
# =================
#
# Train a neural network based on the feature and target data that has
# been saved to file, for a specified number of training cases. The
# network is tagged (using the environment variable $(TAG)) to distinguish
# the model and its outputs.
train:
	touch timing.log
	d=$$(date +%s) && \
	python3 test_and_train.py -m $(MODEL) -n $(NUM_TRAINING_CASES) --tag $(TAG) && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Training completed in $$(($$(date +%s)-d)) seconds" >> timing.log && \
	echo "" >> timing.log

# Plot loss functions
# ===================
#
# Once the network has been trained, plot the training and validation loss
# curves against iteration count.
plot_progress:
	python3 plot_progress.py $(MODEL) --tag $(TAG)

# Feature importance experiment
# =============================
#
# Perform an experiment that tests how sensitive the trained network is to
# each of its inputs (i.e. the features). If it is particularly sensitive to
# one of the features then we deduce that the feature is in some sense
# "important" to the network.
plot_importance:
	python3 compute_importance.py $(MODEL) $(NUM_TRAINING_CASES) --tag $(TAG)
	python3 plot_importance.py $(MODEL) $(NUM_TRAINING_CASES) --tag $(TAG)

# --- Test the neural network

test: snapshot_go snapshot_ml uniform go ml plot_convergence

# Apply goal-oriented adaptation to the test cases
# ================================================
#
# Apply goal-oriented mesh adaptation to the testing cases, thereby
# generating lots of output data in Paraview format. These include
# the meshes, solution fields, error indicators and metrics.
snapshot_go:
	touch timing.log
	d=$$(date +%s) && \
	for case in $(TESTING_CASES); do \
		for approach in $(APPROACHES); do \
			python3 run_adapt.py $(MODEL) $$case -a $$approach $(PETSC_OPTIONS); \
		done; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Goal-oriented snapshots generated in $$(($$(date +%s)-d)) seconds" >> timing.log
	echo "" >> timing.log

# Apply data-driven adaptation to the test cases
# ==============================================
#
# Apply data-driven adaptation based on the trained network to the testing
# cases, thereby generating lots of output data in Paraview format. These
# include the meshes, solution fields, error indicators and metrics.
snapshot_ml:
	touch timing.log
	d=$$(date +%s) && \
	for case in $(TESTING_CASES); do \
		for approach in $(APPROACHES); do \
			python3 run_adapt_ml.py $(MODEL) $$case -a $$approach --tag $(TAG) $(PETSC_OPTIONS); \
		done; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Data-driven snapshots generated in $$(($$(date +%s)-d)) seconds" >> timing.log
	echo "" >> timing.log

# Convergence analysis for uniform refinement
# ===========================================
#
# Run the model on a sequence of uniformly refined meshes.
uniform:
	touch timing.log
	d=$$(date +%s) && \
	for case in $(TESTING_CASES); do \
		python3 run_uniform_refinement.py $(MODEL) $$case; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Uniform refinement completed in $$(($$(date +%s)-d)) seconds" >> timing.log
	echo "" >> timing.log

# Convergence analysis for goal-oriented adaptation
# =================================================
#
# Run the model with the standard goal-oriented approach for
# a range of target metric complexities.
go:
	touch timing.log
	d=$$(date +%s) && \
	for case in $(TESTING_CASES); do \
		for approach in $(APPROACHES); do \
			python3 run_adaptation_loop.py $(MODEL) $$case -a $$approach $(PETSC_OPTIONS); \
		done; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Goal-oriented adaptation completed in $$(($$(date +%s)-d)) seconds" >> timing.log
	echo "" >> timing.log

# Convergence analysis for data-driven adaptation
# ===============================================
#
# Run the model with the data-driven approach based on the
# trained network for a range of target metric complexities.
ml:
	touch timing.log
	d=$$(date +%s) && \
	for case in $(TESTING_CASES); do \
		for approach in $(APPROACHES); do \
			python3 run_adaptation_loop_ml.py $(MODEL) $$case -a $$approach --tag $(TAG) $(PETSC_OPTIONS); \
		done; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Data-driven adaptation completed in $$(($$(date +%s)-d)) seconds" >> timing.log
	echo "" >> timing.log

# Plot convergence curves
# =======================
#
# Plot the data points generated during the `uniform`, `go` and
# `ml` recipes and annotate with lines of best fit, where appropriate.
plot_convergence:
	for case in $(TESTING_CASES); do \
		python3 plot_convergence.py $(MODEL) $$case --tag $(TAG); \
	done

# --- Profiling experiments

# NOTE: The following recipes are somewhat redundant. Similar information
#       can be obtained from the outputs of the `uniform`, `go` and `ml`
#       recipes by running `plot_timings.py` with the appropriate input
#       parameters.

# Profiling for uniform refinement
# ================================
#
# Run the model on a fine fixed mesh generated by refining the initial
# mesh four times and output the PETSc logging information in a format
# that can be then turned into a flamegraph.
profile_uni:
	touch timing.log
	d=$$(date +%s) && \
	for case in $(TESTING_CASES); do \
		python3 run_fixed_mesh.py $(MODEL) $$case --optimise --num_refinements 4 $(PETSC_OPTIONS) -log_view :logview.txt:ascii_flamegraph; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Uniform refinement profiling run completed in $$(($$(date +%s)-d)) seconds" >> timing.log && \
	echo "" >> timing.log
	for case in $(TESTING_CASES); do \
		flamegraph.pl --title "Uniform refinement ($$case)" logview.txt > $(MODEL)/outputs/$$case/uni.svg && \
		rm logview.txt; \
	done

# Profiling for goal-oriented adaptation
# ======================================
#
# Run the model using the standard goal-oriented approach with a fairly
# high target metric complexity of 64,000 and output the PETSc logging
# information in a format that can be then turned into a flamegraph.
profile_go:
	touch timing.log
	d=$$(date +%s) && \
	for case in $(TESTING_CASES); do \
		python3 run_adapt.py $(MODEL) $$case -a anisotropic --optimise --target_complexity 64000 $(PETSC_OPTIONS) -log_view :logview.txt:ascii_flamegraph; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Goal-oriented adaptation profiling run completed in $$(($$(date +%s)-d)) seconds" >> timing.log
	echo "" >> timing.log
	for case in $(TESTING_CASES); do \
		flamegraph.pl --title "Goal-oriented adaptation ($$case)" logview.txt > $(MODEL)/outputs/$$case/go.svg && \
		rm logview.txt; \
	done

# Profiling for data-driven adaptation
# ====================================
#
# Run the model using the data-driven adaptation approach based on the
# trained network with a fairly high target metric complexity of 64,000
# and output the PETSc logging information in a format that can be then
# turned into a flamegraph.
profile_ml:
	touch timing.log
	d=$$(date +%s) && \
	for case in $(TESTING_CASES); do \
		python3 run_adapt_ml.py $(MODEL) $$case -a anisotropic --optimise --target_complexity 64000 $(PETSC_OPTIONS) --tag all -log_view :logview.txt:ascii_flamegraph; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Data-driven adaptation profiling run completed in $$(($$(date +%s)-d)) seconds" >> timing.log
	echo "" >> timing.log
	for case in $(TESTING_CASES); do \
		flamegraph.pl --title "Data-driven adaptation ($$case)" logview.txt > $(MODEL)/outputs/$$case/ml.svg && \
		rm logview.txt; \
	done
