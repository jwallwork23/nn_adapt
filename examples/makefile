all: setup network convergence profile snapshot

# --- Configurable parameters

MODELS				= turbine
NUM_TRAINING_CASES	= 100
TESTING_CASES		= aligned offset

# --- Parameters that should not need modifying

TRAINING_CASES	= $(shell seq 1 ${NUM_TRAINING_CASES})
CASES			= ${TRAINING_CASES} ${TESTING_CASES}

.PHONY: plot_config, plot_progress, plot_importance, plot_convergence

# --- Setup directories and meshes

setup: dir mesh plot_config

dir:
	for model in $(MODELS); do \
		mkdir -p $$model; \
		cd $$model && touch __init__.py && mkdir -p data outputs plots && cd ..; \
		for test_case in $(TESTING_CASES); do \
			mkdir -p $$model/outputs/$$test_case; \
		done; \
	done

mesh:
	touch timing.log
	d=$$(date +%s) && \
	for model in $(MODELS); do \
		for test_case in $(CASES); do \
			python3 meshgen.py $$model $$test_case; \
			cd $$model/meshes && gmsh -2 $$test_case.geo && cd ../..; \
		done; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Meshes built in $$(($$(date +%s)-d)) seconds" >> timing.log

plot_config:
	for model in $(MODELS); do \
		python3 plot_config.py $$model 'train'; \
		python3 plot_config.py $$model 'test'; \
	done

clean:
	for model in $(MODELS); do \
		cd $$model && rm -rf data outputs plots __pycache__ && cd ..; \
	done

# --- Train the neural network

network: features train plot_progress plot_importance

features:
	touch timing.log
	d=$$(date +%s) && \
	for model in $(MODELS); do \
		for test_case in $(TRAINING_CASES); do \
			python3 run_adapt.py $$model $$test_case -a isotropic --no_outputs; \
			python3 run_adapt.py $$model $$test_case -a anisotropic --no_outputs; \
		done; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Features generated in $$(($$(date +%s)-d)) seconds" >> timing.log
	echo "" >> timing.log

train:
	touch timing.log
	d=$$(date +%s) && \
	for model in $(MODELS); do \
		python3 test_and_train.py -m $$model -n $(NUM_TRAINING_CASES); \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Training completed in $$(($$(date +%s)-d)) seconds" >> timing.log
	echo "" >> timing.log

plot_progress:
	for model in $(MODELS); do \
		python3 plot_progress.py $$model; \
	done

plot_importance:
	for model in $(MODELS); do \
		python3 plot_importance.py $$model $(NUM_TRAINING_CASES); \
	done

# --- Snapshots

snapshot:
	touch timing.log
	d=$$(date +%s) && \
	for model in $(MODELS); do \
		for test_case in $(TESTING_CASES); do \
			python3 run_adapt.py $$model $$test_case -a isotropic; \
			python3 run_adapt.py $$model $$test_case -a anisotropic; \
			python3 run_adapt_ml.py $$model $$test_case -a isotropic; \
			python3 run_adapt_ml.py $$model $$test_case -a anisotropic; \
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
	for model in $(MODELS); do \
		for test_case in $(TESTING_CASES); do \
			python3 run_uniform_refinement.py $$model $$test_case; \
		done; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Uniform refinement completed in $$(($$(date +%s)-d)) seconds" >> timing.log
	echo "" >> timing.log

go:
	touch timing.log
	d=$$(date +%s) && \
	for model in $(MODELS); do \
		for test_case in $(TESTING_CASES); do \
			python3 run_adaptation_loop.py $$model $$test_case -a isotropic; \
			python3 run_adaptation_loop.py $$model $$test_case -a anisotropic; \
		done; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Goal-oriented adaptation completed in $$(($$(date +%s)-d)) seconds" >> timing.log
	echo "" >> timing.log

ml:
	touch timing.log
	d=$$(date +%s) && \
	for model in $(MODELS); do \
		for test_case in $(TESTING_CASES); do \
			python3 run_adaptation_loop_ml.py $$model $$test_case -a isotropic; \
			python3 run_adaptation_loop_ml.py $$model $$test_case -a anisotropic; \
		done; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Data-driven adaptation completed in $$(($$(date +%s)-d)) seconds" >> timing.log
	echo "" >> timing.log

plot_convergence:
	for model in $(MODELS); do \
		for test_case in $(TESTING_CASES); do \
			python3 plot_convergence.py $$model $$test_case; \
		done; \
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
	for model in $(MODELS); do \
		for test_case in $(TESTING_CASES); do \
			python3 run_fixed_mesh.py $$model $$test_case --optimise --num_refinements 4 -log_view :$$model/outputs/$$test_case/logview_uni.txt:ascii_flamegraph; \
		done; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Uniform refinement profiling run completed in $$(($$(date +%s)-d)) seconds" >> timing.log && \
	echo "" >> timing.log
	cd $$model/outputs/$$test_case && \
	flamegraph.pl --title "Uniform refinement ($$test_case)" logview.txt > uni.svg && \
	rm logview.txt && cd ../../..

profile_go:
	touch timing.log
	d=$$(date +%s) && \
	for model in $(MODELS); do \
		for test_case in $(TESTING_CASES); do \
			python3 run_adapt.py $$model $$test_case -a anisotropic --optimise --target_complexity 64000 -log_view :$$model/outputs/$$test_case/logview.txt:ascii_flamegraph; \
		done; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Goal-oriented adaptation profiling run completed in $$(($$(date +%s)-d)) seconds" >> timing.log && \
	echo "" >> timing.log
	cd $$model/outputs/$$test_case && \
	flamegraph.pl --title "Goal-oriented adaptation ($$test_case)" logview.txt > go.svg && \
	rm logview.txt && ../../..

profile_ml:
	touch timing.log
	d=$$(date +%s) && \
	for model in $(MODELS); do \
		for test_case in $(TESTING_CASES); do \
			python3 run_adapt_ml.py $$model $$test_case -a anisotropic --optimise --target_complexity 64000 -log_view :$$model/outputs/$$test_case/logview.txt:ascii_flamegraph; \
		done; \
	done && \
	date >> timing.log && \
	git log -n 1 --oneline >> timing.log && \
	echo "Data-driven adaptation profiling run completed in $$(($$(date +%s)-d)) seconds" >> timing.log && \
	echo "" >> timing.log
	cd $$model/outputs/$$test_case && \
	flamegraph.pl --title "Data-driven adaptation ($$test_case)" logview.txt > ml.svg && \
	rm logview.txt && cd ../../..
