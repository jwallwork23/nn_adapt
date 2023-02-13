"""
Run a given ``test_case`` of a ``model`` using goal-oriented
mesh adaptation in a fixed point iteration loop, for a sequence
of increasing target metric complexities,
"""
from nn_adapt.features import *
from nn_adapt.parse import Parser, positive_float
from nn_adapt.metric import *
from nn_adapt.solving import *
from nn_adapt.utility import ConvergenceTracker
from firedrake.meshadapt import adapt

import importlib
import numpy as np
from time import perf_counter


set_log_level(ERROR)

# Parse user input
parser = Parser("run_adaptation_loop.py")
parser.parse_num_refinements(default=24)
parser.parse_approach()
parser.parse_convergence_criteria()
parser.parse_target_complexity()
parser.add_argument(
    "--factor",
    help="Power by which to increase target metric complexity",
    type=positive_float,
    default=0.25,
)
parsed_args, unknown_args = parser.parse_known_args()
model = parsed_args.model
try:
    test_case = int(parsed_args.test_case)
    assert test_case > 0
except ValueError:
    test_case = parsed_args.test_case
approach = parsed_args.approach
num_refinements = parsed_args.num_refinements
base_complexity = parsed_args.base_complexity
f = parsed_args.factor

# Setup
setup = importlib.import_module(f"{model}.config")
setup.initialise(test_case)
unit = setup.parameters.qoi_unit

# Run adaptation loop
qois, dofs, elements, estimators, niter = [], [], [], [], []
components = ("forward", "adjoint", "estimator", "metric", "adapt")
times = []
print(f"Test case {test_case}")
for i in range(num_refinements + 1):
    try:
        target_complexity = 100.0 * 2 ** (f * i)
        kwargs = {
            "enrichment_method": "h",
            "interpolant": "Clement",
            "average": True,
            "anisotropic": approach == "anisotropic",
            "retall": True,
            "h_min": setup.parameters.h_min,
            "h_max": setup.parameters.h_max,
            "a_max": 1.0e5,
        }
        if hasattr(setup, "initial_mesh"):
            mesh = setup.initial_mesh()
        else:
            mesh = Mesh(f"{model}/meshes/{test_case}.msh")
        ct = ConvergenceTracker(mesh, parsed_args)
        print(f"  Target {target_complexity}\n    Mesh 0")
        print(f"      Element count        = {ct.elements_old}")
        times.append(-perf_counter())
        for ct.fp_iteration in range(ct.maxiter + 1):

            # Ramp up the target complexity
            kwargs["target_complexity"] = ramp_complexity(
                base_complexity, target_complexity, ct.fp_iteration
            )

            # Compute goal-oriented metric
            out = go_metric(mesh, setup, convergence_checker=ct, **kwargs)
            qoi = out["qoi"]
            print(f"      Quantity of Interest = {qoi} {unit}")
            if "adjoint" not in out:
                break
            estimator = out["estimator"]
            print(f"      Error estimator      = {estimator}")
            if "metric" not in out:
                break
            fwd_sol, adj_sol = (
                out["forward"],
                out["adjoint"],
            )
            dwr, metric = out["dwr"], out["metric"]
            spaces = [sol[0][0].function_space() for sol in fwd_sol.values()]
            dof = sum(np.array([fs.dof_count for fs in spaces]).flatten())
            print(f"      DoF count            = {dof}")

            # Adapt the mesh
            mesh = adapt(mesh, metric)
            print(f"    Mesh {ct.fp_iteration+1}")
            cells = mesh.num_cells()
            print(f"      Element count        = {cells}")
            if ct.check_elements(cells):
                break
            ct.check_maxiter()
        print(
            f"    Terminated after {ct.fp_iteration+1} iterations due to {ct.converged_reason}"
        )
        times[-1] += perf_counter()
        qois.append(qoi)
        dofs.append(dof)
        elements.append(cells)
        estimators.append(estimator)
        niter.append(ct.fp_iteration + 1)
        np.save(f"{model}/data/qois_GO{approach}_{test_case}", qois)
        np.save(f"{model}/data/dofs_GO{approach}_{test_case}", dofs)
        np.save(f"{model}/data/elements_GO{approach}_{test_case}", elements)
        np.save(f"{model}/data/estimators_GO{approach}_{test_case}", estimators)
        np.save(f"{model}/data/niter_GO{approach}_{test_case}", niter)
        np.save(f"{model}/data/times_all_GO{approach}_{test_case}", times)
    except ConvergenceError:
        print("Skipping due to convergence error")
        continue
