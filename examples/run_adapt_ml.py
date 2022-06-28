"""
Run a given ``test_case`` of a ``model`` using data-driven
mesh adaptation in a fixed point iteration loop.
"""
from nn_adapt.ann import *
from nn_adapt.features import *
from nn_adapt.parse import Parser
from nn_adapt.metric import *
from nn_adapt.solving import *
from nn_adapt.utility import ConvergenceTracker
from firedrake.meshadapt import *

import importlib
from time import perf_counter


# Parse user input
parser = Parser("run_adapt_ml.py")
parser.parse_approach()
parser.parse_convergence_criteria()
parser.parse_preproc()
parser.parse_target_complexity()
parser.parse_tag()
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
preproc = parsed_args.preproc
optimise = parsed_args.optimise
tag = parsed_args.tag
if not optimise:
    from pyroteus.utility import File

# Setup
start_time = perf_counter()
setup = importlib.import_module(f"{model}.config")
setup.initialise(test_case)
unit = setup.parameters.qoi_unit
mesh = Mesh(f"{model}/meshes/{test_case}.msh")

# Load the model
layout = importlib.import_module(f"{model}.network").NetLayout()
nn = SimpleNet(layout).to(device)
nn.load_state_dict(torch.load(f"{model}/model_{tag}.pt"))
nn.eval()

# Run adaptation loop
ct = ConvergenceTracker(mesh, parsed_args)
if not optimise:
    output_dir = f"{model}/outputs/{test_case}/ML/{approach}"
    fwd_file = File(f"{output_dir}/forward.pvd")
    adj_file = File(f"{output_dir}/adjoint.pvd")
    ee_file = File(f"{output_dir}/estimator.pvd")
    metric_file = File(f"{output_dir}/metric.pvd")
    mesh_file = File(f"{output_dir}/mesh.pvd")
    mesh_file.write(mesh.coordinates)
kwargs = {}
print(f"Test case {test_case}")
print("  Mesh 0")
print(f"    Element count        = {ct.elements_old}")
for ct.fp_iteration in range(ct.maxiter + 1):

    # Ramp up the target complexity
    target_ramp = ramp_complexity(base_complexity, target_complexity, ct.fp_iteration)

    # Solve forward and adjoint and compute Hessians
    out = get_solutions(mesh, setup, convergence_checker=ct, **kwargs)
    qoi, fwd_sol = out["qoi"], out["forward"]
    print(f"    Quantity of Interest = {qoi} {unit}")
    dof = sum(fwd_sol.function_space().dof_count)
    print(f"    DoF count            = {dof}")
    if "adjoint" not in out:
        break
    adj_sol = out["adjoint"]
    if not optimise:
        fwd_file.write(*fwd_sol.split())
        adj_file.write(*adj_sol.split())
    P0 = FunctionSpace(mesh, "DG", 0)
    P0_ten = TensorFunctionSpace(mesh, "DG", 0)

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
    with PETSc.Log.Event("Network"):
        features = collect_features(
            extract_features(setup, fwd_sol, adj_sol, preproc=preproc)
        )

        # Run model
        with PETSc.Log.Event("Propagate"):
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
    with PETSc.Log.Event("Error estimation"):
        estimator = dwr.vector().gather().sum()
        print(f"    Error estimator      = {estimator}")
        if ct.check_estimator(estimator):
            break
    if not optimise:
        ee_file.write(dwr)

    # Construct metric
    with PETSc.Log.Event("Metric construction"):
        if approach == "anisotropic":
            hessian = combine_metrics(*get_hessians(fwd_sol), average=False)
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
        space_normalise(p1metric, target_ramp, "inf")
        enforce_element_constraints(
            p1metric, setup.parameters.h_min, setup.parameters.h_max, 1.0e05
        )

        # Adapt the mesh and check for element count convergence
        metric = RiemannianMetric(mesh)
        metric.assign(p1metric)
    if not optimise:
        metric_file.write(p0metric)

    # Adapt the mesh and check for element count convergence
    with PETSc.Log.Event("Mesh adaptation"):
        mesh = adapt(mesh, metric)
    if not optimise:
        mesh_file.write(mesh.coordinates)
    elements = mesh.num_cells()
    print(f"  Mesh {ct.fp_iteration+1}")
    print(f"    Element count        = {elements}")
    if ct.check_elements(elements):
        break
    ct.check_maxiter()
print(f"  Terminated after {ct.fp_iteration+1} iterations due to {ct.converged_reason}")
print(f"  Total time taken: {perf_counter() - start_time:.2f} seconds")
