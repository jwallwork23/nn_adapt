from nn_adapt import *
import argparse
import importlib
import numpy as np


set_log_level(ERROR)

# Parse for test case and number of refinements
parser = argparse.ArgumentParser()
parser.add_argument('test_case', help='The configuration file number')
parser.add_argument('-num_refinements', help='Number of refinements to consider (default 3)')
parser.add_argument('-miniter', help='Minimum number of iterations (default 3)')
parser.add_argument('-maxiter', help='Maximum number of iterations (default 35)')
parser.add_argument('-qoi_rtol', help='Relative tolerance for QoI (default 0.005)')
parser.add_argument('-element_rtol', help='Relative tolerance for element count (default 0.005)')
parser.add_argument('-estimator_rtol', help='Relative tolerance for error estimator (default 0.005)')
parsed_args, unknown_args = parser.parse_known_args()
test_case = int(parsed_args.test_case)
assert test_case in [0, 1, 2, 3, 4]
num_refinements = int(parsed_args.num_refinements or 3)
assert num_refinements > 0
miniter = int(parsed_args.miniter or 3)
assert miniter >= 0
maxiter = int(parsed_args.maxiter or 35)
assert maxiter >= miniter
qoi_rtol = float(parsed_args.qoi_rtol or 0.005)
assert qoi_rtol > 0.0
element_rtol = float(parsed_args.element_rtol or 0.005)
assert element_rtol > 0.0
estimator_rtol = float(parsed_args.estimator_rtol or 0.005)
assert estimator_rtol > 0.0
# p = 1

# Setup
config = importlib.import_module(f'config{test_case}')
mesh = config.mesh
field = config.fields[0]

# Run adaptation loop
qois = []
dofs = []
elements = []
estimators = []
for i in range(num_refinements+1):
    target_complexity = 1000.0*4**i
    kwargs = {
        'enrichment_method': 'h',
        'target_complexity': target_complexity,
        'average': True,
        'retall': True,
    }
    qoi_old = None
    elements_old = mesh.num_cells()
    estimator_old = None
    converged_reason = None
    print(f'Target {target_complexity}\n  Mesh 0')
    print(f'    Element count        = {elements_old}')
    for fp_iteration in range(maxiter+1):

        # Compute goal-oriented metric
        p0metric, dwr, fwd_sol, adj_sol, dwr_plus, adj_sol_plus, mesh_seq = go_metric(mesh, config, **kwargs)

        # Check for QoI convergence
        qoi = mesh_seq.J
        print(f'    Quantity of Interest = {qoi}')
        if qoi_old is not None and fp_iteration >= miniter:
            if abs(qoi - qoi_old) < qoi_rtol*abs(qoi_old):
                converged_reason = 'QoI convergence'
                break
        qoi_old = qoi

        # Check for error estimator convergence
        estimator = dwr.vector().gather().sum()
        print(f'    Error estimator      = {estimator}')
        if estimator_old is not None and fp_iteration >= miniter:
            if abs(estimator - estimator_old) < estimator_rtol*abs(estimator_old):
                converged_reason = 'error estimator convergence'
                break
        estimator_old = estimator

        # Process metric
        P1_ten = TensorFunctionSpace(mesh, 'CG', 1)
        p1metric = hessian_metric(clement_interpolant(p0metric))
        # space_normalise(p1metric, target_complexity, p)
        enforce_element_constraints(p1metric, 1.0e-05, 10.0, 1.0e+05)

        # Adapt the mesh and check for element count convergence
        mesh = adapt(mesh, p1metric)
        print(f'  Mesh {fp_iteration+1}')
        print(f'    Element count        = {elements}')
        if fp_iteration >= miniter:
            if abs(mesh.num_cells() - elements_old) < element_rtol*abs(elements_old):
                converged_reason = 'element count convergence'
                break
        elements_old = mesh.num_cells()

        # Check for reaching maximum number of iterations
        if fp_iteration == maxiter:
            converged_reason = 'reaching maximum iteration count'
    print(f'  Terminated after {fp_iteration+1} iterations due to {converged_reason}')
    qois.append(qoi)
    dofs.append(sum(fwd_sol.function_space().dof_count))
    elements.append(elements_old)
    estimators.append(estimator)
    np.save(f'data/qois_go{test_case}', qois)
    np.save(f'data/dofs_go{test_case}', dofs)
    np.save(f'data/elements_go{test_case}', elements)
    np.save(f'data/estimators_go{test_case}', estimators)
