"""
Run a given ``test_case`` of a ``model`` using data-driven
mesh adaptation in a fixed point iteration loop, for a sequence
of increasing target metric complexities,
"""
from nn_adapt.ann import *
from nn_adapt.features import *
from nn_adapt.parse import Parser
from nn_adapt.metric import *
from nn_adapt.solving import *
from firedrake.meshadapt import *

import git
import importlib
import numpy as np
from time import perf_counter


set_log_level(ERROR)

# Parse user input
parser = Parser("run_adaptation_loop_ml.py")
parser.parse_num_refinements(default=5)
parser.parse_approach()
parser.parse_convergence_criteria()
parser.parse_tag()
parser.parse_preproc()
parsed_args, unknown_args = parser.parse_known_args()
model = parsed_args.model
try:
    test_case = int(parsed_args.test_case)
    assert test_case > 0
except ValueError:
    test_case = parsed_args.test_case
approach = parsed_args.approach
num_refinements = parsed_args.num_refinements
miniter = parsed_args.miniter
maxiter = parsed_args.maxiter
assert maxiter >= miniter
qoi_rtol = parsed_args.qoi_rtol
element_rtol = parsed_args.element_rtol
estimator_rtol = parsed_args.estimator_rtol
preproc = parsed_args.preproc
tag = parsed_args.tag or git.Repo(search_parent_directories=True).head.object.hexsha

# Setup
setup = importlib.import_module(f"{model}.config")
setup.initialise(test_case)
unit = setup.parameters.qoi_unit

# Load the model
layout = importlib.import_module(f"{model}.network").NetLayout()
nn = SimpleNet(layout).to(device)
nn.load_state_dict(torch.load(f"{model}/model_{tag}.pt"))
nn.eval()

# Run adaptation loop
qois = []
dofs = []
elements = []
times = []
print(f"Test case {test_case}")
for i in range(num_refinements + 1):
    target_complexity = 200.0 * 4**i
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
        target_ramp = ramp_complexity(200.0, target_complexity, fp_iteration)

        # Solve forward and adjoint and compute Hessians
        fwd_sol, adj_sol = get_solutions(mesh, setup)
        P0 = FunctionSpace(mesh, "DG", 0)
        P0_ten = TensorFunctionSpace(mesh, "DG", 0)

        # Check for QoI convergence
        qoi = assemble(setup.get_qoi(mesh)(fwd_sol))
        print(f"      Quantity of Interest = {qoi} {unit}")
        if qoi_old is not None and fp_iteration >= miniter:
            if abs(qoi - qoi_old) < qoi_rtol * abs(qoi_old):
                converged_reason = "QoI convergence"
                break
        qoi_old = qoi

        # Extract features
        features = collect_features(
            extract_features(setup, fwd_sol, adj_sol, preproc=preproc)
        )

        # Run model
        test_targets = np.array([])
        with torch.no_grad():
            for i in range(features.shape[0]):
                test_x = torch.Tensor(features[i]).to(device)
                test_prediction = nn(test_x)
                test_targets = np.concatenate(
                    (test_targets, np.array(test_prediction.cpu()))
                )
        dwr = Function(P0)
        dwr.dat.data[:] = np.abs(test_targets)

        # Check for error estimator convergence
        estimator = dwr.vector().gather().sum()
        print(f"      Error estimator      = {estimator}")
        if estimator_old is not None and fp_iteration >= miniter:
            if abs(estimator - estimator_old) < estimator_rtol * abs(estimator_old):
                converged_reason = "error estimator convergence"
                break
        estimator_old = estimator

        # Construct metric
        if approach == "anisotropic":
            hessian = combine_metrics(*get_hessians(adj_sol), average=False)
        else:
            hessian = None
        p0metric = anisotropic_metric(
            dwr,
            hessian,
            target_complexity=target_ramp,
            target_space=P0_ten,
            interpolant="L2",
        )

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
    dofs.append(sum(fwd_sol.function_space().dof_count))
    elements.append(elements_old)
    np.save(f"{model}/data/qois_ML{approach}_{test_case}", qois)
    np.save(f"{model}/data/dofs_ML{approach}_{test_case}", dofs)
    np.save(f"{model}/data/elements_ML{approach}_{test_case}", elements)
