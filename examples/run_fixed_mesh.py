"""
Run a given ``test_case`` of a ``model`` on the initial mesh alone.
"""
from nn_adapt.parse import Parser
from nn_adapt.solving import *
from thetis import print_output
from firedrake.petsc import PETSc
import importlib
import os
from time import perf_counter


start_time = perf_counter()

# Parse user input
parser = Parser("run_fixed_mesh.py")
parser.parse_num_refinements(default=0)
parsed_args, unknown_args = parser.parse_known_args()
model = parsed_args.model
try:
    test_case = int(parsed_args.test_case)
    assert test_case > 0
except ValueError:
    test_case = parsed_args.test_case

# Setup
setup = importlib.import_module(f"{model}.config")
setup.initialise(test_case)
unit = setup.parameters.qoi_unit
mesh = Mesh(f"{model}/meshes/{test_case}.msh")
if parsed_args.num_refinements > 0:
    with PETSc.Log.Event("Hierarchy"):
        mesh = MeshHierarchy(mesh, parsed_args.num_refinements)[-1]

# Solve and evaluate QoI
out = get_solutions(mesh, setup, solve_adjoint=not parsed_args.optimise)
qoi = out["qoi"]
print_output(f"QoI for test case {test_case} = {qoi:.8f} {unit}")
if not parsed_args.optimise:
    File(f"{model}/outputs/{test_case}/fixed/forward.pvd").write(
        *out["forward"].split()
    )
    File(f"{model}/outputs/{test_case}/fixed/adjoint.pvd").write(
        *out["adjoint"].split()
    )
print_output(f"  Total time taken: {perf_counter() - start_time:.2f} seconds")
