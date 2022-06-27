### Examples

Currently, only one model has been implemented: steady-state flow through a tidal turbine
solved using the shallow water equations using [Thetis]. A mixed discontinuous-continuous
finite element method is used -- P1DG for velocity and P2 for free surface elevation.

Adding a new model `foo` would require a few things:

* A `models/foo.py` model description file, which should mimic `models/turbine.py` and include all the same functions for setting up and solving the problem.
* A problem configuration file `foo/config.py`, which reads from `models/foo.py` and initialises parameter classes as required.
* A `foo/meshgen.py` file for generating gmsh geometry files associated with the initial meshes, driven by a function ``generate_geo``.
* A `foo/network.py` file for setting up the neural network architecture.
* A `foo/plotting.py` file for specifying how the problem setups should be plotted and variables should be separated by importance.
* Add test cases in a file `foo/testing_cases.txt`.
* Set the `MODEL` environment variable in the makefile to `foo`.
* You can also modify `NUM_TRAINING_CASES` as desired.

The entire workflow for using goal-oriented mesh adaptation to generate feature data,
training the network and testing it should all be possible using the `makefile` recipes.
There are also various plotting and profiling routines.

[Thetis]: https://thetisproject.org/ "Thetis"
