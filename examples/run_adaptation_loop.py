"""
Run a given ``test_case`` of a ``model`` using goal-oriented
mesh adaptation in a fixed point iteration loop, for a sequence
of increasing target metric complexities,
"""
from nn_adapt.features import *
from nn_adapt.metric import *
from nn_adapt.solving import *

import argparse
import importlib
import numpy as np
from time import perf_counter


set_log_level(ERROR)

# Parse for test case and number of refinements
parser = argparse.ArgumentParser(prog="run_adaptation_loop.py")
parser.add_argument("model", help="The model")
parser.add_argument("test_case", help="The configuration file number")
parser.add_argument("-anisotropic", help="Toggle isotropic vs. anisotropic metric")
parser.add_argument("-num_refinements", help="Number of refinements (default 4)")
parser.add_argument("-miniter", help="Minimum number of iterations (default 3)")
parser.add_argument("-maxiter", help="Maximum number of iterations (default 35)")
parser.add_argument("-qoi_rtol", help="Relative tolerance for QoI (default 0.001)")
parser.add_argument("-element_rtol", help="Element count tolerance (default 0.001)")
parser.add_argument("-estimator_rtol", help="Error estimator tolerance (default 0.001)")
parsed_args, unknown_args = parser.parse_known_args()
model = parsed_args.model
try:
    test_case = int(parsed_args.test_case)
    assert test_case > 0
except ValueError:
    test_case = parsed_args.test_case
approach = "isotropic" if parsed_args.anisotropic in [None, "0"] else "anisotropic"
num_refinements = int(parsed_args.num_refinements or 5)
assert num_refinements > 0
miniter = int(parsed_args.miniter or 3)
assert miniter >= 0
maxiter = int(parsed_args.maxiter or 35)
assert maxiter >= miniter
qoi_rtol = float(parsed_args.qoi_rtol or 0.001)
assert qoi_rtol > 0.0
element_rtol = float(parsed_args.element_rtol or 0.001)
assert element_rtol > 0.0
estimator_rtol = float(parsed_args.estimator_rtol or 0.001)
assert estimator_rtol > 0.0

# Setup
setup = importlib.import_module(f"{model}.config")
setup.initialise(test_case)
unit = setup.parameters.qoi_unit

# Run adaptation loop
qois = []
dofs = []
elements = []
estimators = []
times = []
print(f"Test case {test_case}")
for i in range(num_refinements + 1):
    target_complexity = 200.0 * 4**i
    kwargs = {
        "enrichment_method": "h",
        "average": False,
        "anisotropic": approach == "anisotropic",
        "retall": True,
    }
    mesh = Mesh(f"{model}/meshes/{test_case}.msh")
    qoi_old = None
    elements_old = mesh.num_cells()
    estimator_old = None
    converged_reason = None
    print(f"  Target {target_complexity}\n    Mesh 0")
    print(f"      Element count        = {elements_old}")
    cpu_timestamp = perf_counter()
    for fp_iteration in range(maxiter + 1):

        # Ramp up the target complexity
        kwargs["target_complexity"] = ramp_complexity(
            250.0, target_complexity, fp_iteration
        )

        # Compute goal-oriented metric
        p0metric, dwr, fwd_sol, adj_sol = go_metric(mesh, setup, **kwargs)
        dof = sum(fwd_sol.function_space().dof_count)
        print(f"      DoF count            = {dof}")

        # Check for QoI convergence
        qoi = assemble(setup.get_qoi(mesh)(fwd_sol))
        print(f"      Quantity of Interest = {qoi} {unit}")
        if qoi_old is not None and fp_iteration >= miniter:
            if abs(qoi - qoi_old) < qoi_rtol * abs(qoi_old):
                converged_reason = "QoI convergence"
                break
        qoi_old = qoi

        # Check for error estimator convergence
        estimator = dwr.vector().gather().sum()
        print(f"      Error estimator      = {estimator}")
        if estimator_old is not None and fp_iteration >= miniter:
            if abs(estimator - estimator_old) < estimator_rtol * abs(estimator_old):
                converged_reason = "error estimator convergence"
                break
        estimator_old = estimator

        # Process metric
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        p1metric = hessian_metric(clement_interpolant(p0metric))
        enforce_element_constraints(
            p1metric, setup.parameters.h_min, setup.parameters.h_max, 1.0e05
        )

        # Adapt the mesh and check for element count convergence
        metric = RiemannianMetric(mesh)
        metric.assign(p1metric)
        mesh = adapt(mesh, metric)
        print(f"    Mesh {fp_iteration+1}")
        if fp_iteration >= miniter:
            if abs(mesh.num_cells() - elements_old) < element_rtol * abs(elements_old):
                converged_reason = "element count convergence"
                break
        elements_old = mesh.num_cells()
        print(f"      Element count        = {elements_old}")

        # Check for reaching maximum number of iterations
        if fp_iteration == maxiter:
            converged_reason = "reaching maximum iteration count"
    print(f"    Terminated after {fp_iteration+1} iterations due to {converged_reason}")
    times.append(perf_counter() - cpu_timestamp)
    qois.append(qoi)
    dofs.append(dof)
    elements.append(elements_old)
    estimators.append(estimator)
    np.save(f"{model}/data/qois_GO{approach}_{test_case}", qois)
    np.save(f"{model}/data/dofs_GO{approach}_{test_case}", dofs)
    np.save(f"{model}/data/elements_GO{approach}_{test_case}", elements)
    np.save(f"{model}/data/estimators_GO{approach}_{test_case}", estimators)
