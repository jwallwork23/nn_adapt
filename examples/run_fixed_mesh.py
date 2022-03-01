"""
Run a given ``test_case`` of a ``model`` on the initial mesh alone.
"""
from nn_adapt.parse import *
from nn_adapt.solving import *
from firedrake.petsc import PETSc
import importlib
from time import perf_counter


start_time = perf_counter()

# Parse for test case
parser = argparse.ArgumentParser(
    prog="run_fixed_mesh.py",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("model", help="The model", type=str)
parser.add_argument("test_case", help="The configuration file number or name")
parser.add_argument("--num_refinements", help="Number of mesh refinements", type=positive_int, default=0)
parser.add_argument("--optimise", help="Turn off plotting and debugging", action="store_true")
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
sols = get_solutions(mesh, setup, solve_adjoint=not parsed_args.optimise)
msg = f"QoI for test case {test_case}"
if parsed_args.optimise:
    print(
        f"{msg} = {assemble(setup.get_qoi(mesh)(sols)):.2f} {unit}"
    )
else:
    print(
        f"{msg} = {assemble(setup.get_qoi(mesh)(sols[0])):.2f} {unit}"
    )
    File(f"{model}/outputs/{test_case}/fixed/forward.pvd").write(*sols[0].split())
    File(f"{model}/outputs/{test_case}/fixed/adjoint.pvd").write(*sols[1].split())
print(f"  Total time taken: {perf_counter() - start_time:.2f} seconds")
