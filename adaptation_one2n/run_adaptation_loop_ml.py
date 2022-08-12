"""
Run a given ``test_case`` of a ``model`` using data-driven
mesh adaptation in a fixed point iteration loop, for a sequence
of increasing target metric complexities,
"""
from nn_adapt.ann import *
from nn_adapt.features import *
from nn_adapt.parse import Parser, positive_float
from nn_adapt.metric_one2n import *
from nn_adapt.solving_one2n import *
from nn_adapt.utility import ConvergenceTracker
from firedrake.meshadapt import *

import importlib
import numpy as np
from time import perf_counter


set_log_level(ERROR)

# Parse user input
parser = Parser("run_adaptation_loop_ml.py")
parser.parse_num_refinements(default=24)
parser.parse_approach()
parser.parse_convergence_criteria()
parser.parse_preproc()
parser.parse_tag()
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
preproc = parsed_args.preproc
tag = parsed_args.tag
base_complexity = parsed_args.base_complexity
f = parsed_args.factor

# Setup
setup = importlib.import_module(f"{model}.config")
setup.initialise(test_case)
unit = setup.parameters.qoi_unit

# Load the model
layout = importlib.import_module(f"{model}.network").NetLayout()
nn = SingleLayerFCNN(layout, preproc=preproc).to(device)
nn.load_state_dict(torch.load(f"{model}/model_{tag}.pt"))
nn.eval()

# Run adaptation loop
qois, dofs, elements, estimators, niter = [], [], [], [], []
components = ("forward", "adjoint", "estimator", "metric", "adapt")
times = {c: [] for c in components}
times["all"] = []
print(f"Test case {test_case}")
for i in range(num_refinements + 1):
    try:
        target_complexity = 100.0 * 2 ** (f * i)
        if hasattr(setup, "initial_mesh"):
            mesh = setup.initial_mesh
        else:
            mesh = Mesh(f"{model}/meshes/{test_case}.msh")
        ct = ConvergenceTracker(mesh, parsed_args)
        kwargs = {}
        print(f"  Target {target_complexity}\n    Mesh 0")
        print(f"      Element count        = {ct.elements_old}")
        times["all"].append(-perf_counter())
        for c in components:
            times[c].append(0.0)
        for ct.fp_iteration in range(ct.maxiter + 1):

            # Ramp up the target complexity
            target_ramp = ramp_complexity(
                base_complexity, target_complexity, ct.fp_iteration
            )

            # Solve forward and adjoint and compute Hessians
            out = get_solutions_one2n(mesh, setup, convergence_checker=ct, **kwargs)
            qoi = out["qoi"]
            times["forward"][-1] += out["times"]["forward"]
            print(f"      Quantity of Interest = {qoi} {unit}")
            if "adjoint" not in out:
                break
            times["adjoint"][-1] += out["times"]["adjoint"]
            fwd_sol, adj_sol = out["forward"], out["adjoint"]
            dof = sum(np.array([fwd_sol[0].function_space().dof_count]).flatten())
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

            # Extract features
            out["times"]["estimator"] = -perf_counter()
            fwd_sol_integrate = time_integrate(fwd_sol)
            adj_sol_integrate = time_integrate(adj_sol)
            features = extract_features(setup, fwd_sol_integrate, adj_sol_integrate)
            features = collect_features_sample(features, layout)

            # Run model
            test_targets = np.array([])
            with torch.no_grad():
                for i in range(features.shape[0]):
                    test_x = torch.Tensor(features[i]).to(device)
                    test_prediction = nn(test_x)
                    test_targets = np.concatenate(
                        (test_targets, np.array(test_prediction.cpu()))
                    )
            P0 = FunctionSpace(mesh, "DG", 0)
            dwr = Function(P0)
            dwr.dat.data[:] = np.abs(test_targets)

            # Check for error estimator convergence
            estimator = dwr.vector().gather().sum()
            out["times"]["estimator"] += perf_counter()
            times["estimator"][-1] += out["times"]["estimator"]
            print(f"      Error estimator      = {estimator}")
            if ct.check_estimator(estimator):
                break

            # Construct metric
            out["times"]["metric"] = -perf_counter()
            if approach == "anisotropic":
                hessian = combine_metrics(*get_hessians(fwd_sol_integrate), average=True)
            else:
                hessian = None
            P1_ten = TensorFunctionSpace(mesh, "CG", 1)
            M = anisotropic_metric(
                dwr,
                hessian=hessian,
                target_complexity=target_ramp,
                target_space=P1_ten,
                interpolant="Clement",
            )
            space_normalise(M, target_ramp, "inf")
            enforce_element_constraints(
                M, setup.parameters.h_min, setup.parameters.h_max, 1.0e05
            )
            metric = RiemannianMetric(mesh)
            metric.assign(M)
            out["times"]["metric"] += perf_counter()
            times["metric"][-1] += out["times"]["metric"]

            # Adapt the mesh and check for element count convergence
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
        np.save(f"{model}/data/qois_ML{approach}_{test_case}_{tag}", qois)
        np.save(f"{model}/data/dofs_ML{approach}_{test_case}_{tag}", dofs)
        np.save(f"{model}/data/elements_ML{approach}_{test_case}_{tag}", elements)
        np.save(f"{model}/data/estimators_ML{approach}_{test_case}_{tag}", estimators)
        np.save(f"{model}/data/niter_ML{approach}_{test_case}_{tag}", niter)
        np.save(f"{model}/data/times_all_ML{approach}_{test_case}_{tag}", times["all"])
        for c in components:
            np.save(f"{model}/data/times_{c}_ML{approach}_{test_case}_{tag}", times[c])
    except ConvergenceError:
        print("Skipping due to convergence error")
        continue
