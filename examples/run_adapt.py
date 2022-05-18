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
from firedrake.meshadapt import *
from firedrake.petsc import PETSc

import importlib
import numpy as np
from time import perf_counter


start_time = perf_counter()
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
miniter = parsed_args.miniter
maxiter = parsed_args.maxiter
assert maxiter >= miniter
qoi_rtol = parsed_args.qoi_rtol
element_rtol = parsed_args.element_rtol
estimator_rtol = parsed_args.estimator_rtol
target_complexity = parsed_args.target_complexity
optimise = parsed_args.optimise
no_outputs = parsed_args.no_outputs or optimise
if not no_outputs:
    from pyroteus.utility import File

# Setup
setup = importlib.import_module(f"{model}.config")
setup.initialise(test_case)
unit = setup.parameters.qoi_unit
mesh = Mesh(f"{model}/meshes/{test_case}.msh")
dim = mesh.topological_dimension()
Nd = dim**2

# Run adaptation loop
kwargs = {
    "enrichment_method": "h",
    "average": False,
    "anisotropic": approach == "anisotropic",
    "retall": True,
}
qoi_old = None
elements_old = mesh.num_cells()
estimator_old = None
converged_reason = None
if not no_outputs:
    output_dir = f"{model}/outputs/{test_case}/GO/{approach}"
    fwd_file = File(f"{output_dir}/forward.pvd")
    adj_file = File(f"{output_dir}/adjoint.pvd")
    ee_file = File(f"{output_dir}/estimator.pvd")
    metric_file = File(f"{output_dir}/metric.pvd")
print(f"Test case {test_case}")
print("  Mesh 0")
print(f"    Element count        = {elements_old}")
data_dir = f"{model}/data"
for fp_iteration in range(maxiter + 1):
    suffix = f"{test_case}_GO{approach}_{fp_iteration}"

    # Ramp up the target complexity
    kwargs["target_complexity"] = ramp_complexity(
        200.0, target_complexity, fp_iteration
    )

    # Compute goal-oriented metric
    p0metric, dwr, fwd_sol, adj_sol = go_metric(mesh, setup, **kwargs)
    dof = sum(fwd_sol.function_space().dof_count)
    print(f"    DoF count            = {dof}")
    if not no_outputs:
        fwd_file.write(*fwd_sol.split())
        adj_file.write(*adj_sol.split())
        ee_file.write(dwr)
        metric_file.write(p0metric)

    # Extract features
    if not optimise:
        features = extract_features(setup, fwd_sol, adj_sol)
        target = dwr.dat.data.flatten()
        assert not np.isnan(target).any()
        for key, value in features.items():
            np.save(f"{data_dir}/feature_{key}_{suffix}", value)
        np.save(f"{data_dir}/target_{suffix}", target)

    # Check for QoI convergence
    qoi = assemble(setup.get_qoi(mesh)(fwd_sol))
    print(f"    Quantity of Interest = {qoi} {unit}")
    if qoi_old is not None and fp_iteration >= miniter:
        if abs(qoi - qoi_old) < qoi_rtol * abs(qoi_old):
            converged_reason = "QoI convergence"
            break
    qoi_old = qoi

    # Check for error estimator convergence
    with PETSc.Log.Event("Error estimation"):
        P0 = dwr.function_space()
        estimator = dwr.vector().gather().sum()
        print(f"    Error estimator      = {estimator}")
        if estimator_old is not None and fp_iteration >= miniter:
            if abs(estimator - estimator_old) < estimator_rtol * abs(estimator_old):
                converged_reason = "error estimator convergence"
                break
        estimator_old = estimator

    # Process metric
    with PETSc.Log.Event("Metric construction"):
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        p1metric = hessian_metric(clement_interpolant(p0metric))
        space_normalise(p1metric, target_ramp, "inf")
        enforce_element_constraints(
            p1metric, setup.parameters.h_min, setup.parameters.h_max, 1.0e05
        )
        metric = RiemannianMetric(mesh)
        metric.assign(p1metric)

    # Adapt the mesh and check for element count convergence
    with PETSc.Log.Event("Mesh adaptation"):
        mesh = adapt(mesh, metric)
    elements = mesh.num_cells()
    print(f"  Mesh {fp_iteration+1}")
    print(f"    Element count        = {elements}")
    if fp_iteration >= miniter:
        if abs(elements - elements_old) < element_rtol * abs(elements_old):
            converged_reason = "element count convergence"
            break
    elements_old = elements

    # Check for reaching maximum number of iterations
    if fp_iteration == maxiter:
        converged_reason = "reaching maximum iteration count"
print(f"  Terminated after {fp_iteration+1} iterations due to {converged_reason}")
print(f"  Total time taken: {perf_counter() - start_time:.2f} seconds")
