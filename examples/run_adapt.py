from nn_adapt.features import *
from nn_adapt.metric import *
from nn_adapt.solving import *
from firedrake.petsc import PETSc

import argparse
import importlib
import numpy as np
import os
from time import perf_counter


start_time = perf_counter()
set_log_level(ERROR)

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
parser.add_argument('-optimise', help='Turn off plotting and debugging (default False)')
parsed_args, unknown_args = parser.parse_known_args()
model = parsed_args.model
assert model in ['stokes', 'turbine']
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
optimise = bool(parsed_args.optimise or False)
if not optimise:
    from pyroteus.utility import File

# Setup
setup = importlib.import_module(f'{model}.config{test_case}')
field = setup.fields[0]
if model == 'stokes':
    plex = PETSc.DMPlex().create()
    plex.createFromFile(f'{os.path.abspath(os.path.dirname(__file__))}/{model}/meshes/{test_case}.h5')
    mesh = Mesh(plex)
else:
    mesh = Mesh(f'{os.path.abspath(os.path.dirname(__file__))}/{model}/meshes/{test_case}.msh')
dim = mesh.topological_dimension()
Nd = dim**2

# Run adaptation loop
kwargs = {
    'enrichment_method': 'h',
    'average': True,
    'anisotropic': approach == 'anisotropic',
    'retall': True,
}
qoi_old = None
elements_old = mesh.num_cells()
estimator_old = None
converged_reason = None
if not optimise:
    fwd_file = File(f'{model}/outputs/GO/{approach}/forward{test_case}.pvd')
    adj_file = File(f'{model}/outputs/GO/{approach}/adjoint{test_case}.pvd')
    adj_plus_file = File(f'{model}/outputs/GO/{approach}/enriched_adjoint{test_case}.pvd')
    ee_file = File(f'{model}/outputs/GO/{approach}/estimator{test_case}.pvd')
    ee_plus_file = File(f'{model}/outputs/GO/{approach}/enriched_estimator{test_case}.pvd')
    metric_file = File(f'{model}/outputs/GO/{approach}/metric{test_case}.pvd')
print(f'Test case {test_case}')
print('  Mesh 0')
print(f'    Element count        = {elements_old}')
for fp_iteration in range(maxiter+1):

    # Ramp up the target complexity
    kwargs['target_complexity'] = ramp_complexity(250.0, target_complexity, fp_iteration)

    # Compute goal-oriented metric
    p0metric, dwr, fwd_sol, adj_sol, dwr_plus, adj_sol_plus = go_metric(mesh, setup, **kwargs)
    dof = sum(fwd_sol.function_space().dof_count)
    print(f'    DoF count            = {dof}')
    if not optimise:
        fwd_file.write(*fwd_sol.split())
        adj_file.write(*adj_sol.split())
        adj_plus_file.write(*adj_sol_plus.split())
        ee_file.write(dwr)
        ee_plus_file.write(dwr_plus)
        metric_file.write(p0metric)

    # Extract features
    if not optimise:
        features = extract_features(setup, fwd_sol, adj_sol)
        targets = dwr.dat.data.flatten()
        assert not np.isnan(targets).any()
        np.save(f'{model}/data/features{test_case}_GO{approach}_{fp_iteration}', features)
        np.save(f'{model}/data/targets{test_case}_GO{approach}_{fp_iteration}', targets)

    # Check for QoI convergence
    qoi = assemble(setup.get_qoi(mesh)(fwd_sol))
    print(f'    Quantity of Interest = {qoi}')
    if qoi_old is not None and fp_iteration >= miniter:
        if abs(qoi - qoi_old) < qoi_rtol*abs(qoi_old):
            converged_reason = 'QoI convergence'
            break
    qoi_old = qoi

    # Check for error estimator convergence
    with PETSc.Log.Event('Error estimation'):
        P0 = dwr.function_space()
        estimator = dwr.vector().gather().sum()
        print(f'    Error estimator      = {estimator}')
        if estimator_old is not None and fp_iteration >= miniter:
            if abs(estimator - estimator_old) < estimator_rtol*abs(estimator_old):
                converged_reason = 'error estimator convergence'
                break
        estimator_old = estimator

    # Process metric
    with PETSc.Log.Event('Metric construction'):
        P1_ten = TensorFunctionSpace(mesh, 'CG', 1)
        p1metric = hessian_metric(clement_interpolant(p0metric))
        enforce_element_constraints(p1metric,
                                    setup.parameters.h_min,
                                    setup.parameters.h_max,
                                    1.0e+05)
        metric = RiemannianMetric(mesh)
        metric.assign(p1metric)

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
