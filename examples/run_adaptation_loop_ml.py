from nn_adapt import *
from nn_adapt.ann import *
import argparse
import importlib
import numpy as np
from time import perf_counter


set_log_level(ERROR)

# Parse for test case and number of refinements
parser = argparse.ArgumentParser()
parser.add_argument('model', help='The model')
parser.add_argument('test_case', help='The configuration file number')
parser.add_argument('-num_refinements', help='Number of refinements to consider (default 3)')
parser.add_argument('-miniter', help='Minimum number of iterations (default 3)')
parser.add_argument('-maxiter', help='Maximum number of iterations (default 35)')
parser.add_argument('-qoi_rtol', help='Relative tolerance for QoI (default 0.001)')
parser.add_argument('-element_rtol', help='Relative tolerance for element count (default 0.005)')
parser.add_argument('-norm_order', help='Metric normalisation order (default 1.0)')
parser.add_argument('-preproc', help='Function for preprocessing data (default "arctan")')
parsed_args, unknown_args = parser.parse_known_args()
model = parsed_args.model
assert model in ['stokes']
test_case = int(parsed_args.test_case)
assert test_case in [0, 1, 2, 3, 4]
num_refinements = int(parsed_args.num_refinements or 4)
assert num_refinements > 0
miniter = int(parsed_args.miniter or 3)
assert miniter >= 0
maxiter = int(parsed_args.maxiter or 35)
assert maxiter >= miniter
qoi_rtol = float(parsed_args.qoi_rtol or 0.001)
assert qoi_rtol > 0.0
element_rtol = float(parsed_args.element_rtol or 0.005)
assert element_rtol > 0.0
p = float(parsed_args.norm_order or 1.0)
assert p >= 1.0
preproc = parsed_args.preproc or 'arctan'
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

# Load the model
nn = SimpleNet().to(device)
nn.load_state_dict(torch.load(f'{model}/model.pt'))
nn.eval()

# Run adaptation loop
qois = []
dofs = []
elements = []
times = []
print(f'Test case {test_case}')
for i in range(num_refinements+1):
    target_complexity = 250.0*4**i
    kwargs = {
        'enrichment_method': 'h',
        'target_complexity': target_complexity,
        'average': True,
        'retall': True,
    }
    plex = PETSc.DMPlex().create()
    plex.createFromFile(f'{os.path.abspath(os.path.dirname(__file__))}/{model}/meshes/{test_case}.h5')
    mesh = Mesh(plex)
    dim = mesh.topological_dimension()
    Nd = dim**2
    qoi_old = None
    elements_old = mesh.num_cells()
    converged_reason = None
    print(f'  Target {target_complexity}\n    Mesh 0')
    print(f'      Element count        = {elements_old}')
    cpu_timestamp = perf_counter()
    for fp_iteration in range(maxiter+1):

        # Solve forward and adjoint and compute Hessians
        fwd_sol, adj_sol, mesh_seq = get_solutions(mesh, config)
        hessians = [*get_hessians(fwd_sol), *get_hessians(adj_sol)]

        # Check for QoI convergence
        qoi = mesh_seq.J
        print(f'      Quantity of Interest = {qoi}')
        if qoi_old is not None and fp_iteration >= miniter:
            if abs(qoi - qoi_old) < qoi_rtol*abs(qoi_old):
                converged_reason = 'QoI convergence'
                break
        qoi_old = qoi

        # Extract features
        num_inputs = config.parameters.num_inputs
        features = np.array([]).reshape(0, num_inputs)
        targets = np.array([]).reshape(0, 3)
        errors = np.array([])
        ar = get_aspect_ratios2d(mesh)
        h = interpolate(CellSize(mesh), ar.function_space()).dat.data
        ar = ar.dat.data
        bnd_nodes = DirichletBC(mesh.coordinates.function_space(), 0, 'on_boundary').nodes
        bnd_tags = [1 if node in bnd_nodes else 0 for node in range(elements_old)]
        Re = config.parameters.Re(mesh)  # TODO: Mesh Reynolds number
        shape = (elements_old, 3, Nd)
        indices = [i*dim + j for i in range(dim) for j in range(i, dim)]
        hessians = [np.reshape(get_values_at_elements(H).dat.data, shape)[:, :, indices] for H in hessians]
        for i in range(elements_old):
            feature = np.concatenate((*[H[i].flatten() for H in hessians], [ar[i], h[i], bnd_tags[i], Re]))
            features = np.concatenate((features, feature.reshape(1, num_inputs)))

        # Preprocess features
        shape = features.shape
        if preproc != 'none':
            features = f(features.reshape(1, shape[0]*shape[1])).reshape(*shape)

        # Run model
        test_targets = np.array([])
        with torch.no_grad():
            for i in range(features.shape[0]):
                test_x = torch.Tensor(features[i]).to(device)
                test_prediction = nn(test_x)
                test_targets = np.concatenate((test_targets, np.array(test_prediction.cpu())))

        # Extract metric
        test_targets = test_targets.reshape(test_targets.shape[0]//3, 3)
        M = np.c_[test_targets, np.ones(test_targets.shape[0])]
        M[:, 3] = M[:, 2]
        M[:, 2] = M[:, 1]
        P0_ten = TensorFunctionSpace(mesh, 'DG', 0)
        p0metric = Function(P0_ten)
        p0metric.dat.data[:] = M.reshape(p0metric.dat.data.shape)

        # Process metric
        P1_ten = TensorFunctionSpace(mesh, 'CG', 1)
        p1metric = hessian_metric(clement_interpolant(p0metric))
        space_normalise(p1metric, target_complexity, p)
        enforce_element_constraints(p1metric, 1.0e-05, 10.0, 1.0e+05)

        # Adapt the mesh and check for element count convergence
        mesh = adapt(mesh, p1metric)
        print(f'    Mesh {fp_iteration+1}')
        if fp_iteration >= miniter:
            if abs(mesh.num_cells() - elements_old) < element_rtol*abs(elements_old):
                converged_reason = 'element count convergence'
                break
        elements_old = mesh.num_cells()
        print(f'      Element count        = {elements_old}')

        # Check for reaching maximum number of iterations
        if fp_iteration == maxiter:
            converged_reason = 'reaching maximum iteration count'
    print(f'    Terminated after {fp_iteration+1} iterations due to {converged_reason}')
    times.append(perf_counter() - cpu_timestamp)
    qois.append(qoi)
    dofs.append(sum(fwd_sol.function_space().dof_count))
    elements.append(elements_old)
    np.save(f'{model}/data/qois_ml{test_case}', qois)
    np.save(f'{model}/data/dofs_ml{test_case}', dofs)
    np.save(f'{model}/data/elements_ml{test_case}', elements)
