from nn_adapt import *
from nn_adapt.ann import *
from firedrake.petsc import PETSc
import argparse
import importlib


set_log_level(ERROR)

# Parse for test case and number of refinements
parser = argparse.ArgumentParser()
parser.add_argument('model', help='The model')
parser.add_argument('test_case', help='The configuration file number')
parser.add_argument('-miniter', help='Minimum number of iterations (default 3)')
parser.add_argument('-maxiter', help='Maximum number of iterations (default 35)')
parser.add_argument('-qoi_rtol', help='Relative tolerance for QoI (default 0.001)')
parser.add_argument('-element_rtol', help='Relative tolerance for element count (default 0.005)')
parser.add_argument('-target_complexity', help='Target metric complexity (default 4000.0)')
parser.add_argument('-norm_order', help='Metric normalisation order (default 1.0)')
parser.add_argument('-preproc', help='Function for preprocessing data (default "none")')
parsed_args, unknown_args = parser.parse_known_args()
model = parsed_args.model
assert model in ['stokes']
test_case = int(parsed_args.test_case)
assert test_case in [0, 1, 2, 3, 4]
miniter = int(parsed_args.miniter or 3)
assert miniter >= 0
maxiter = int(parsed_args.maxiter or 35)
assert maxiter >= miniter
qoi_rtol = float(parsed_args.qoi_rtol or 0.001)
assert qoi_rtol > 0.0
element_rtol = float(parsed_args.element_rtol or 0.005)
assert element_rtol > 0.0
target_complexity = float(parsed_args.target_complexity or 4000.0)
assert target_complexity > 0.0
p = float(parsed_args.norm_order or 1.0)
assert p >= 1.0
preproc = parsed_args.preproc or 'none'
if preproc == 'arctan':
    f = np.arctan
elif preproc == 'tanh':
    f = np.tanh
elif preproc == 'logabs':
    f = lambda x: np.ln(np.abs(x))
elif preproc != 'none':
    raise ValueError(f'Preprocessor "{preproc}" not recognised.')

# Setup
config = importlib.import_module(f'{model}.config{test_case}')
field = config.fields[0]
plex = PETSc.DMPlex().create()
plex.createFromFile(f'{os.path.abspath(os.path.dirname(__file__))}/{model}/meshes/{test_case}.h5')
mesh = Mesh(plex)
dim = mesh.topological_dimension()
Nd = dim**2
num_inputs = config.parameters.num_inputs

# Load the model
nn = SimpleNet().to(device)
nn.load_state_dict(torch.load(f'{model}/model.pt'))
nn.eval()

# Run adaptation loop
kwargs = {
    'enrichment_method': 'h',
    'target_complexity': target_complexity,
    'average': True,
    'retall': True,
}
qoi_old = None
elements_old = mesh.num_cells()
converged_reason = None
fwd_file = File(f'{model}/outputs/ml/forward{test_case}.pvd')
adj_file = File(f'{model}/outputs/ml/adjoint{test_case}.pvd')
metric_file = File(f'{model}/outputs/ml/metric{test_case}.pvd')
print(f'Test case {test_case}')
print('  Mesh 0')
print(f'    Element count        = {elements_old}')
for fp_iteration in range(maxiter+1):
    features = np.array([]).reshape(0, num_inputs)

    # Solve forward and adjoint and compute Hessians
    fwd_sol, adj_sol, mesh_seq = get_solutions(mesh, config)
    dof = sum(fwd_sol.function_space().dof_count)
    print(f'    DoF count            = {dof}')
    hessians = [*get_hessians(fwd_sol), *get_hessians(adj_sol)]
    fwd_file.write(*fwd_sol.split())
    adj_file.write(*adj_sol.split())
    P0 = FunctionSpace(mesh, 'DG', 0)
    P0_ten = TensorFunctionSpace(mesh, 'DG', 0)

    # Check for QoI convergence
    qoi = mesh_seq.J
    print(f'    Quantity of Interest = {qoi}')
    if qoi_old is not None and fp_iteration >= miniter:
        if abs(qoi - qoi_old) < qoi_rtol*abs(qoi_old):
            converged_reason = 'QoI convergence'
            break
    qoi_old = qoi

    # Extract features
    features = extract_features(config, fwd_sol, hessians)
    with PETSc.Log.Event('nn_adapt.preprocess_features'):
        shape = features.shape
        if preproc != 'none':
            features = f(features.reshape(1, shape[0]*shape[1])).reshape(*shape)

    # Run model
    with PETSc.Log.Event('nn_adapt.run_model'):
        test_targets = np.array([])
        with torch.no_grad():
            for i in range(features.shape[0]):
                test_x = torch.Tensor(features[i]).to(device)
                test_prediction = nn(test_x)
                test_targets = np.concatenate((test_targets, np.array(test_prediction.cpu())))

    # Extract metric
    with PETSc.Log.Event('nn_adapt.extract_metric'):
        test_targets = test_targets.reshape(test_targets.shape[0]//3, 3)
        M = np.c_[test_targets, np.ones(test_targets.shape[0])]
        M[:, 3] = M[:, 2]
        M[:, 2] = M[:, 1]
        p0metric = Function(P0_ten)
        p0metric.dat.data[:] = M.reshape(p0metric.dat.data.shape)
        metric_file.write(p0metric)

    # Process metric
    P1_ten = TensorFunctionSpace(mesh, 'CG', 1)
    p1metric = hessian_metric(clement_interpolant(p0metric))
    space_normalise(p1metric, target_complexity, p)
    enforce_element_constraints(p1metric, 1.0e-05, 10.0, 1.0e+05)

    # Adapt the mesh and check for element count convergence
    mesh = adapt(mesh, p1metric)
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
