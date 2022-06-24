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
from firedrake.meshadapt import *

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
times = {c: [] for c in components}
times["all"] = []
print(f"Test case {test_case}")
for i in range(num_refinements + 1):
    try:
        target_complexity = 100.0 * 2 ** (f * i)
        kwargs = {
            "enrichment_method": "h",
            "average": False,
            "anisotropic": approach == "anisotropic",
            "retall": True,
        }
        mesh = Mesh(f"{model}/meshes/{test_case}.msh")
        ct = ConvergenceTracker(mesh, parsed_args)
        print(f"  Target {target_complexity}\n    Mesh 0")
        print(f"      Element count        = {ct.elements_old}")
        times["all"].append(-perf_counter())
        for c in components:
            times[c].append(0.0)
        for ct.fp_iteration in range(ct.maxiter + 1):

            # Ramp up the target complexity
            target_ramp = ramp_complexity(base_complexity, target_complexity, ct.fp_iteration)
            kwargs["target_complexity"] = target_ramp

            # Compute goal-oriented metric
            out = go_metric(mesh, setup, convergence_checker=ct, **kwargs)
            qoi = out["qoi"]
            times["forward"][-1] += out["times"]["forward"]
            print(f"      Quantity of Interest = {qoi} {unit}")
            if "adjoint" not in out:
                break
            estimator = out["estimator"]
            times["adjoint"][-1] += out["times"]["adjoint"]
            times["estimator"][-1] += out["times"]["estimator"]
            print(f"      Error estimator      = {estimator}")
            if "metric" not in out:
                break
            fwd_sol, adj_sol = out["forward"], out["adjoint"],
            dwr, p0metric = out["dwr"], out["metric"]
            dof = sum(fwd_sol.function_space().dof_count)
            print(f"      DoF count            = {dof}")

            def proj(V):
                """
                After the first iteration, project the previous
                solution as the initial guess.
                """
                ic = Function(V)
                try:
                    ic.project(fwd_sol)
                except NotImplementedError:
                    for c_init, c in zip(ic.split(), fwd_sol.split()):
                        c_init.project(c)
                return ic

            # Use previous solution for initial guess
            if parsed_args.transfer:
                kwargs["init"] = proj

            # Process metric
            out["times"]["metric"] -= perf_counter()
            P1_ten = TensorFunctionSpace(mesh, "CG", 1)
            p1metric = hessian_metric(clement_interpolant(p0metric))
            space_normalise(p1metric, target_ramp, "inf")
            enforce_element_constraints(
                p1metric, setup.parameters.h_min, setup.parameters.h_max, 1.0e05
            )

            # Adapt the mesh and check for element count convergence
            metric = RiemannianMetric(mesh)
            metric.assign(p1metric)
            out["times"]["metric"] += perf_counter()
            times["metric"][-1] += out["times"]["metric"]
            out["times"]["adapt"] = -perf_counter()
            mesh = adapt(mesh, metric)
            out["times"]["adapt"] += perf_counter()
            times["adapt"][-1] += out["times"]["adapt"]
            print(f"    Mesh {ct.fp_iteration+1}")
            cells = mesh.num_cells()
            print(f"      Element count        = {cells}")
            if ct.check_elements(cells):
                break
            ct.check_maxiter()
        print(
            f"    Terminated after {ct.fp_iteration+1} iterations due to {ct.converged_reason}"
        )
        times["all"][-1] += perf_counter()
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
        np.save(f"{model}/data/times_all_GO{approach}_{test_case}", times["all"])
        for c in components:
            np.save(f"{model}/data/times_{c}_GO{approach}_{test_case}", times[c])
    except ConvergenceError:
        print("Skipping due to convergence error")
        continue
