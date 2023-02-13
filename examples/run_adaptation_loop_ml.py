"""
Run a given ``test_case`` of a ``model`` using data-driven
mesh adaptation in a fixed point iteration loop, for a sequence
of increasing target metric complexities,
"""
from nn_adapt.ann import *
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
times = []
print(f"Test case {test_case}")
for i in range(num_refinements + 1):
    try:
        target_complexity = 100.0 * 2 ** (f * i)
        if hasattr(setup, "initial_mesh"):
            mesh = setup.initial_mesh()
        else:
            mesh = Mesh(f"{model}/meshes/{test_case}.msh")
        ct = ConvergenceTracker(mesh, parsed_args)
        kwargs = {}
        print(f"  Target {target_complexity}\n    Mesh 0")
        print(f"      Element count        = {ct.elements_old}")
        times.append(-perf_counter())
        for ct.fp_iteration in range(ct.maxiter + 1):

            # Ramp up the target complexity
            target_ramp = ramp_complexity(
                base_complexity, target_complexity, ct.fp_iteration
            )

            # Solve forward and adjoint and compute Hessians
            out = get_solutions(mesh, setup, convergence_checker=ct, **kwargs)
            qoi = out["qoi"]
            print(f"      Quantity of Interest = {qoi} {unit}")
            if "adjoint" not in out:
                break
            fwd_sol, adj_sol = out["forward"], out["adjoint"]
            spaces = [sol[0][0].function_space() for sol in fwd_sol.values()]
            dof = sum(np.array([fs.dof_count for fs in spaces]).flatten())
            print(f"      DoF count            = {dof}")

            # Extract features
            field = list(fwd_sol.keys())[0]  # FIXME: Only uses 0th field
            features = extract_features(setup, fwd_sol[field][0][0], adj_sol[field][0][0])  # FIXME
            features = collect_features(features, layout)

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
            # FIXME: Only produces one error indicator

            # Check for error estimator convergence
            estimator = dwr.vector().gather().sum()
            print(f"      Error estimator      = {estimator}")
            if ct.check_estimator(estimator):
                break

            # Construct metric
            if approach == "anisotropic":
                field = list(fwd_sol.keys())[0]
                fwd = fwd_sol[field][0]  # FIXME: Only uses 0th
                hessians = sum([get_hessians(sol) for sol in fwd], start=())
                hessian = combine_metrics(*hessians, average=True)
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

            # Adapt the mesh and check for element count convergence
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
        np.save(f"{model}/data/qois_ML{approach}_{test_case}_{tag}", qois)
        np.save(f"{model}/data/dofs_ML{approach}_{test_case}_{tag}", dofs)
        np.save(f"{model}/data/elements_ML{approach}_{test_case}_{tag}", elements)
        np.save(f"{model}/data/estimators_ML{approach}_{test_case}_{tag}", estimators)
        np.save(f"{model}/data/niter_ML{approach}_{test_case}_{tag}", niter)
        np.save(f"{model}/data/times_all_ML{approach}_{test_case}_{tag}", times)
    except ConvergenceError:
        print("Skipping due to convergence error")
        continue
