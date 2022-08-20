"""
Run a given ``test_case`` of a ``model`` using goal-oriented
mesh adaptation in a fixed point iteration loop.

This is the script where feature data is harvested to train
the neural network on.
"""
from nn_adapt.features import *
from nn_adapt.metric import *
from nn_adapt.parse import Parser
from nn_adapt.solving import *
from nn_adapt.utility import ConvergenceTracker
from firedrake.meshadapt import adapt
from firedrake.petsc import PETSc

import importlib
import numpy as np
from time import perf_counter


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
no_outputs = 1
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
    
try:
    num_subinterval = len(mesh)
except:
    num_subinterval = 1

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
ct = ConvergenceTracker(mesh[0], parsed_args)
if not no_outputs:
    output_dir = f"{model}/outputs/{test_case}/GO/{approach}"
    fwd_file = File(f"{output_dir}/forward.pvd")
    adj_file = File(f"{output_dir}/adjoint.pvd")
    ee_file = File(f"{output_dir}/estimator.pvd")
    metric_file = File(f"{output_dir}/metric.pvd")
    mesh_file = File(f"{output_dir}/mesh.pvd")
    # mesh_file.write(mesh.coordinates)
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
    out = go_metric(mesh, setup, convergence_checker=ct, **kwargs)
    qoi, fwd_sol = out["qoi"], out["forward"]
    print(f"    Quantity of Interest = {qoi} {unit}")
    spaces = [sol[0][0].function_space() for sol in fwd_sol.values()]
    dof = sum(np.array([fs.dof_count for fs in spaces]).flatten())
    print(f"    DoF count            = {dof}")
    if "adjoint" not in out:
        break
    estimator = out["estimator"]
    print(f"    Error estimator      = {estimator}")
    if "metric" not in out:
        break
    adj_sol, dwr, metric = out["adjoint"], out["dwr"], out["metric"]
    if not no_outputs:
        fields = ()
        for sol in fwd_sol.values():
            fields += sol[0][0].split()  # FIXME: Only uses 0th
        fwd_file.write(*fields)
        fields = ()
        for sol in adj_sol.values():
            fields += sol[0][0].split()  # FIXME: Only uses 0th
        adj_file.write(*fields)
        ee_file.write(dwr[0])  # FIXME: Only uses 0th
        metric_file.write(metric.function)

    # Extract features
    if not optimise:
        field = list(fwd_sol.keys())[0]  # FIXME: Only uses 0th field
        features = extract_features(setup, fwd_sol[field][0][0], adj_sol[field][0][0])  # FIXME
        target = dwr[0].dat.data.flatten()  # FIXME: Only uses 0th
        assert not np.isnan(target).any()
        for key, value in features.items():
            np.save(f"{data_dir}/feature_{key}_{suffix}", value)
        np.save(f"{data_dir}/target_{suffix}", target)

    # Adapt the mesh and check for element count convergence
    with PETSc.Log.Event("Mesh adaptation"):
        if num_subinterval == 1:
            mesh = adapt(mesh, metric)
        else:
            for id in range(num_subinterval):
                mesh[id] = adapt(mesh[id], metric[id])
    if not no_outputs:
        mesh_file.write(mesh.coordinates)
        
    if num_subinterval == 1:
        elements = mesh.num_cells()
        print(f"  Mesh {ct.fp_iteration+1}")
        print(f"    Element count        = {elements}")
    else:
        elements_list = np.array([mesh_i.num_cells() for mesh_i in mesh])
        elements = elements_list.mean()
        print(f"  Mesh {ct.fp_iteration+1}")
        print(f"    Element list        = {elements_list}")
    if ct.check_elements(elements):
        break
    ct.check_maxiter()
print(f"  Terminated after {ct.fp_iteration+1} iterations due to {ct.converged_reason}")
print(f"  Total time taken: {perf_counter() - start_time:.2f} seconds")
