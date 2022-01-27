from nn_adapt.ann import *
from nn_adapt.features import *
from nn_adapt.metric import *
from nn_adapt.solving import *

import argparse
import importlib
from time import perf_counter


start_time = perf_counter()

# Parse for test case and number of refinements
parser = argparse.ArgumentParser()
parser.add_argument('model', help='The model')
parser.add_argument('test_case', help='The setupuration file number')
parser.add_argument('-anisotropic', help='Toggle isotropic vs. anisotropic metric')
parser.add_argument('-miniter', help='Minimum number of iterations (default 3)')
parser.add_argument('-maxiter', help='Maximum number of iterations (default 35)')
parser.add_argument('-qoi_rtol', help='Relative tolerance for QoI (default 0.001)')
parser.add_argument('-element_rtol', help='Relative tolerance for element count (default 0.005)')
parser.add_argument('-estimator_rtol', help='Relative tolerance for error estimator (default 0.005)')
parser.add_argument('-target_complexity', help='Target metric complexity (default 4000.0)')
parser.add_argument('-preproc', help='Function for preprocessing data (default "arctan")')
parser.add_argument('-optimise', help='Turn off plotting and debugging (default False)')
parsed_args, unknown_args = parser.parse_known_args()
model = parsed_args.model
assert model in ['turbine']
test_case = int(parsed_args.test_case)
assert test_case in list(range(16))
approach = 'isotropic' if parsed_args.anisotropic in [None, '0'] else 'anisotropic'
miniter = int(parsed_args.miniter or 3)
assert miniter >= 0
maxiter = int(parsed_args.maxiter or 35)
assert maxiter >= miniter
qoi_rtol = float(parsed_args.qoi_rtol or 0.001)
assert qoi_rtol > 0.0
element_rtol = float(parsed_args.element_rtol or 0.005)
assert element_rtol > 0.0
estimator_rtol = float(parsed_args.estimator_rtol or 0.005)
assert estimator_rtol > 0.0
target_complexity = float(parsed_args.target_complexity or 4000.0)
assert target_complexity > 0.0
preproc = parsed_args.preproc or 'arctan'
optimise = bool(parsed_args.optimise or False)
if not optimise:
    from pyroteus.utility import File

# Setup
setup = importlib.import_module(f'{model}.config')
setup.initialise(test_case)
mesh = Mesh(f'{model}/meshes/{test_case}.msh')

# Load the model
layout = importlib.import_module(f'{model}.network').NetLayout()
nn = SimpleNet(layout).to(device)
nn.load_state_dict(torch.load(f'{model}/model.pt'))
nn.eval()

# Run adaptation loop
qoi_old = None
elements_old = mesh.num_cells()
estimator_old = None
converged_reason = None
if not optimise:
    fwd_file = File(f'{model}/outputs/ML/{approach}/forward{test_case}.pvd')
    adj_file = File(f'{model}/outputs/ML/{approach}/adjoint{test_case}.pvd')
    ee_file = File(f'{model}/outputs/ML/{approach}/estimator{test_case}.pvd')
    metric_file = File(f'{model}/outputs/ML/{approach}/metric{test_case}.pvd')
print(f'Test case {test_case}')
print('  Mesh 0')
print(f'    Element count        = {elements_old}')
for fp_iteration in range(maxiter+1):

    # Ramp up the target complexity
    target_ramp = ramp_complexity(250.0, target_complexity, fp_iteration)

    # Solve forward and adjoint and compute Hessians
    fwd_sol, adj_sol = get_solutions(mesh, setup)
    dof = sum(fwd_sol.function_space().dof_count)
    print(f'    DoF count            = {dof}')
    if not optimise:
        fwd_file.write(*fwd_sol.split())
        adj_file.write(*adj_sol.split())
    P0 = FunctionSpace(mesh, 'DG', 0)
    P0_ten = TensorFunctionSpace(mesh, 'DG', 0)

    # Check for QoI convergence
    qoi = assemble(setup.get_qoi(mesh)(fwd_sol))
    print(f'    Quantity of Interest = {qoi}')
    if qoi_old is not None and fp_iteration >= miniter:
        if abs(qoi - qoi_old) < qoi_rtol*abs(qoi_old):
            converged_reason = 'QoI convergence'
            break
    qoi_old = qoi

    # Extract features
    features = extract_features(setup, fwd_sol, adj_sol, preproc=preproc)

    # Run model
    with PETSc.Log.Event('Network propagate'):
        test_targets = np.array([])
        with torch.no_grad():
            for i in range(features.shape[0]):
                test_x = torch.Tensor(features[i]).to(device)
                test_prediction = nn(test_x)
                test_targets = np.concatenate((test_targets, np.array(test_prediction.cpu())))
        dwr = Function(P0)
        dwr.dat.data[:] = np.abs(test_targets)

    # Check for error estimator convergence
    with PETSc.Log.Event('Error estimation'):
        estimator = dwr.vector().gather().sum()
        print(f'    Error estimator      = {estimator}')
        if estimator_old is not None and fp_iteration >= miniter:
            if abs(estimator - estimator_old) < estimator_rtol*abs(estimator_old):
                converged_reason = 'error estimator convergence'
                break
        estimator_old = estimator
    if not optimise:
        ee_file.write(dwr)

    # Construct metric
    with PETSc.Log.Event('Metric construction'):
        if approach == 'anisotropic':
            hessian = combine_metrics(*get_hessians(adj_sol), average=True)
        else:
            hessian = None
        p0metric = anisotropic_metric(
            dwr, hessian, target_complexity=target_ramp,
            target_space=P0_ten, interpolant='L2'
        )

        # Process metric
        P1_ten = TensorFunctionSpace(mesh, 'CG', 1)
        p1metric = hessian_metric(clement_interpolant(p0metric))
        enforce_element_constraints(p1metric,
                                    setup.parameters.h_min,
                                    setup.parameters.h_max,
                                    1.0e+05)

        # Adapt the mesh and check for element count convergence
        metric = RiemannianMetric(mesh)
        metric.assign(p1metric)
    if not optimise:
        metric_file.write(p0metric)

    # Adapt the mesh and check for element count convergence
    with PETSc.Log.Event('Mesh adaptation'):
        mesh = adapt(mesh, metric)
    elements = mesh.num_cells()
    print(f'  Mesh {fp_iteration+1}')
    print(f'    Element count        = {elements}')
    if fp_iteration >= miniter:
        if abs(elements - elements_old) < element_rtol*abs(elements_old):
            converged_reason = 'element count convergence'
            break
    elements_old = elements

    # Check for reaching maximum number of iterations
    if fp_iteration == maxiter:
        converged_reason = 'reaching maximum iteration count'
print(f'  Terminated after {fp_iteration+1} iterations due to {converged_reason}')
print(f'  Total time taken: {perf_counter() - start_time:.2f} seconds')
