import firedrake
from firedrake.petsc import PETSc
from firedrake import op2
import numpy as np
from nn_adapt.solving import split_into_scalars
from pyroteus.metric import *
import ufl


__all__ = ['extract_features', 'get_values_at_elements', 'preprocess_features']


@PETSc.Log.EventDecorator('nn_adapt.extract_components')
def extract_components(matrix):
    r"""
    Extract components of a matrix that describe its
    size, orientation and shape.

    The latter two components are combined in such
    a way that we avoid errors relating to arguments
    zero and :math:`2\pi` being equal.
    """
    density, quotients, evecs = density_and_quotients(matrix, reorder=True)
    fs = density.function_space()
    ar = firedrake.interpolate(ufl.sqrt(quotients[1]), fs)
    armin = ar.vector().gather().min()
    assert armin >= 1.0, f'An element has aspect ratio is less than one ({armin})'
    theta = firedrake.interpolate(ufl.atan(evecs[1, 1]/evecs[1, 0]), fs)
    # return density.dat.data, ar.dat.data, theta.dat.data
    h1 = firedrake.interpolate(ufl.cos(theta)**2/ar + ufl.sin(theta)**2*ar, fs)
    h2 = firedrake.interpolate((1/ar - ar)*ufl.sin(theta)*ufl.cos(theta), fs)
    return density.dat.data, h1.dat.data, h2.dat.data
    # H = firedrake.interpolate(ufl.as_matrix([[h1, h2], [h2, h1]]), matrix.function_space())
    # V, Lambda = compute_eigendecomposition(H)
    # lmin = Lambda.vector().gather().min()
    # assert lmin > 0.0, f'Normalised Hessian is not positive-definite (min eigenvalue {lmin})'
    # Lambda.dat.data[:] = np.log(Lambda.dat.data)
    # logH = assemble_eigendecomposition(V, Lambda)
    # return density.dat.data, logH.dat.data[:, 0, 0], logH.dat.data[:, 0, 1]


@PETSc.Log.EventDecorator("nn_adapt.get_values_at_elements")
def get_values_at_elements(M):
    r"""
    Extract the values for all degrees of freedom associated
    with each element.

    :arg M: a :math:`\mathbb P1` metric :class:`Function`
    :return: a vector :class:`Function` holding all DoFs
    """
    fs = M.function_space()
    mesh = fs.mesh()
    dim = mesh.topological_dimension()
    if dim == 2:
        assert fs.ufl_element().cell() == ufl.triangle, 'Simplex meshes only'
    elif dim == 3:
        assert fs.ufl_element().cell() == ufl.tetrahedron, 'Simplex meshes only'
    else:
        raise ValueError(f'Dimension {dim} not supported')
    el = fs.ufl_element()
    if el.sub_elements() == []:
        size = (dim+1)*el.value_size()*el.degree()
    else:
        size = (dim+1)*sum(sel.value_size()*sel.degree() for sel in el.sub_elements())
    P0_vec = firedrake.VectorFunctionSpace(mesh, 'DG', 0, dim=size)
    values = firedrake.Function(P0_vec)
    kernel = "for (int i=0; i < vertexwise.dofs; i++) elementwise[i] += vertexwise[i];"
    keys = {'vertexwise': (M, op2.READ), 'elementwise': (values, op2.INC)}
    firedrake.par_loop(kernel, ufl.dx, keys)
    return values


@PETSc.Log.EventDecorator('nn_adapt.extract_features')
def extract_features(config, fwd_sol, adj_sol, preproc='none'):
    """
    Extract features from the outputs of a run.

    :arg config: the configuration file
    :arg fwd_sol: the forward solution
    :arg adj_sol: the adjoint solution
    :kwarg preproc: preprocessor function
    """
    mesh = fwd_sol.function_space().mesh()
    P0 = firedrake.FunctionSpace(mesh, 'DG', 0)
    P0_ten = firedrake.TensorFunctionSpace(mesh, 'DG', 0)

    # Mesh Reynolds number
    Re = config.parameters.Re(fwd_sol).dat.data

    # Features describing the mesh element
    J = ufl.Jacobian(mesh)
    JTJ = firedrake.interpolate(ufl.dot(ufl.transpose(J), J), P0_ten)
    d, h1, h2 = extract_components(JTJ)
    bnd_nodes = firedrake.DirichletBC(P0, 0, 'on_boundary').nodes
    bnd_tags = [
        1 if node in bnd_nodes else 0
        for node in range(mesh.num_cells())
    ]

    # Features describing the forward and adjoint solutions
    fwd_sols = split_into_scalars(fwd_sol)
    adj_sols = split_into_scalars(adj_sol)
    sols = sum([fi for i, fi in fwd_sols.items()], start=[]) \
        + sum([ai for i, ai in adj_sols.items()], start=[])
    vals = [get_values_at_elements(s).dat.data for s in sols]

    # Coarse approximation of the error indicator
    dwr = config.dwr_indicator(mesh, fwd_sol, adj_sol)[0].dat.data

    # Combine the features together
    num_inputs = config.parameters.num_inputs
    features = np.array([]).reshape(0, num_inputs)
    for i in range(mesh.num_cells()):
        mesh_features = [d[i], h1[i], h2[i], bnd_tags[i]]
        solution_features = [vv for v in vals for vv in v[i]]
        feature = [Re[i]] + mesh_features + solution_features + [dwr[i]]
        feature = np.reshape(feature, (1, num_inputs))
        features = np.concatenate((features, feature))
    assert not np.isnan(features).any()
    return preprocess_features(features, preproc=preproc)


@PETSc.Log.EventDecorator('nn_adapt.preprocess_features')
def preprocess_features(features, preproc='none'):
    """
    Pre-process features so that they are
    similarly scaled.

    :arg features: the array of features
    :kwarg preproc: preprocessor function
    """
    if preproc == 'none':
        return features
    if preproc == 'arctan':
        f = np.arctan
    elif preproc == 'tanh':
        f = np.tanh
    elif preproc == 'logabs':
        f = lambda x: np.ln(np.abs(x))
    else:
        raise ValueError(f'Preprocessor "{preproc}" not recognised.')
    shape = features.shape
    return f(features.reshape(1, shape[0]*shape[1])).reshape(*shape)
