"""
Run a given ``test_case`` of a ``model`` using goal-oriented
mesh adaptation in a fixed point iteration loop.

This is the script where feature data is harvested to train
the neural network on.
"""
from nn_adapt.features import *
from nn_adapt.metric_one2n import *
from nn_adapt.parse import Parser
from nn_adapt.solving import *
from nn_adapt.solving_one2n import *
from nn_adapt.utility import ConvergenceTracker
from firedrake.meshadapt import adapt
from firedrake.petsc import PETSc

import importlib
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt


set_log_level(ERROR)

# Parse for test case and number of refinements
parser = Parser("run_adapt.py")
parser.parse_approach()
parser.parse_convergence_criteria()
parser.parse_target_complexity()
parser.add_argument("--no_outputs", help="Turn off file outputs", action="store_true")
parsed_args, unknown_args = parser.parse_known_args()
model = parsed_args.model
try:
    test_case = int(parsed_args.test_case)
    assert test_case > 0
except ValueError:
    test_case = parsed_args.test_case
approach = parsed_args.approach
base_complexity = parsed_args.base_complexity
target_complexity = parsed_args.target_complexity
optimise = parsed_args.optimise
no_outputs = parsed_args.no_outputs or optimise
if not no_outputs:
    from pyroteus.utility import File

# Setup
start_time = perf_counter()
setup = importlib.import_module(f"{model}.config")
setup.initialise(test_case)
unit = setup.parameters.qoi_unit
if hasattr(setup, "initial_mesh"):
    mesh = setup.initial_mesh
else:
    mesh = Mesh(f"{model}/meshes/{test_case}.msh")

# Run adaptation loop
kwargs = {
    "interpolant": "Clement",
    "enrichment_method": "h",
    "average": True,
    "anisotropic": approach == "anisotropic",
    "retall": True,
    "h_min": setup.parameters.h_min,
    "h_max": setup.parameters.h_max,
    "a_max": 1.0e5,
}
ct = ConvergenceTracker(mesh, parsed_args)
tt_steps = setup.parameters.tt_steps
if not no_outputs:
    output_dir = f"{model}/outputs/{test_case}/GO/{approach}"
    fwd_file = [File(f"{output_dir}/forward{step}.pvd") for step in range(tt_steps)]
    adj_file = [File(f"{output_dir}/adjoint{step}.pvd") for step in range(tt_steps)]
    ee_file = File(f"{output_dir}/estimator.pvd")
    metric_file = File(f"{output_dir}/metric.pvd")
    mesh_file = File(f"{output_dir}/mesh.pvd")
    mesh_file.write(mesh.coordinates)
print(f"Test case {test_case}")
print("  Mesh 0")
print(f"    Element count        = {ct.elements_old}")
data_dir = f"{model}/data"
for ct.fp_iteration in range(ct.maxiter + 1):
    suffix = f"{test_case}_GO{approach}_{ct.fp_iteration}"

    # Ramp up the target complexity
    kwargs["target_complexity"] = ramp_complexity(
        base_complexity, target_complexity, ct.fp_iteration
    )

    # Compute goal-oriented metric
    out = go_metric_one2n(mesh, setup, convergence_checker=ct, **kwargs)
    qoi, fwd_sol = out["qoi"], out["forward"]
    print(f"    Quantity of Interest = {qoi} {unit}")
    dof = sum(np.array([fwd_sol[0].function_space().dof_count]).flatten())
    print(f"    DoF count            = {dof}")
    if "adjoint" not in out:
        break
    estimator = out["estimator"]
    print(f"    Error estimator      = {estimator}")
    if "metric" not in out:
        break
    adj_sol, dwr, metric = out["adjoint"], out["dwr"], out["metric"]
    
    # fig, axes = plt.subplots(1,2)
    # tricontourf(fwd_sol, axes=axes[0])
    # tricontourf(adj_sol, axes=axes[1])  
    # plt.savefig("out.jpg")
    
    if not no_outputs:
        for step in range(tt_steps):
            fwd_file[step].write(*fwd_sol[step].split())
            adj_file[step].write(*adj_sol[step].split())
        ee_file.write(dwr)
        metric_file.write(metric.function)

    def proj(V):
        """
        After the first iteration, project the previous
        solution as the initial guess.
        """
        ic = Function(V)
        try:
            ic.project(fwd_sol[-1])
        except NotImplementedError:
            for c_init, c in zip(ic.split(), fwd_sol[-1].split()):
                c_init.project(c)
        return ic

    # Use previous solution for initial guess
    if parsed_args.transfer:
        kwargs["init"] = proj

    # # Extract features
    # if not optimise:
    #     features = extract_features(setup, fwd_sol, adj_sol)
    #     target = dwr.dat.data.flatten()
    #     assert not np.isnan(target).any()
    #     for key, value in features.items():
    #         np.save(f"{data_dir}/feature_{key}_{suffix}", value)
    #     np.save(f"{data_dir}/target_{suffix}", target)

    # Adapt the mesh and check for element count convergence
    with PETSc.Log.Event("Mesh adaptation"):
        mesh = adapt(mesh, metric)
    if not no_outputs:
        mesh_file.write(mesh.coordinates)
    elements = mesh.num_cells()
    print(f"  Mesh {ct.fp_iteration+1}")
    print(f"    Element count        = {elements}")
    if ct.check_elements(elements):
        break
    ct.check_maxiter()
print(f"  Terminated after {ct.fp_iteration+1} iterations due to {ct.converged_reason}")
print(f"  Total time taken: {perf_counter() - start_time:.2f} seconds")
