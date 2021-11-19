import firedrake
from firedrake.petsc import PETSc
import numpy as np
from pyroteus.metric import density_and_quotients
import ufl


__all__ = ['extract_features']


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
    yield density.dat.data
    ar = firedrake.interpolate(ufl.sqrt(quotients[0]), fs)
    theta = firedrake.interpolate(ufl.atan(evecs[1, 1]/evecs[1, 0]), fs)
    yield firedrake.interpolate(ufl.cos(theta)**2/ar + ufl.sin(theta)**2*ar, fs).dat.data
    yield firedrake.interpolate((1/ar - ar)*ufl.sin(theta)*ufl.cos(theta), fs).dat.data


@PETSc.Log.EventDecorator('nn_adapt.extract_features')
def extract_features(config, fwd_sol, hessians):
    """
    Extract features from the outputs of a run.

    :arg config: the configuration file
    :arg fwd_sol: the forward solution
    :arg hessians: Hessians of each component of
        the forward and adjoint solutions
    """
    mesh = fwd_sol.function_space().mesh()
    P0 = firedrake.FunctionSpace(mesh, 'DG', 0)
    P0_ten = firedrake.TensorFunctionSpace(mesh, 'DG', 0)
    J = ufl.Jacobian(mesh)
    JTJ = firedrake.interpolate(ufl.dot(ufl.transpose(J), J), P0_ten)
    Re = config.parameters.Re(fwd_sol).dat.data
    bnd_nodes = firedrake.DirichletBC(P0, 0, 'on_boundary').nodes
    bnd_tags = [
        1 if node in bnd_nodes else 0
        for node in range(mesh.num_cells())
    ]
    hessians = [firedrake.interpolate(H, P0_ten) for H in hessians]
    data = sum(
        [list(extract_components(H)) for H in hessians],
        start=[Re, bnd_tags, *extract_components(JTJ)]
    )
    num_inputs = config.parameters.num_inputs
    features = np.array([]).reshape(0, num_inputs)
    for i in range(mesh.num_cells()):
        features = np.concatenate((features, np.reshape([d[i] for d in data], (1, num_inputs))))
    assert not np.isnan(features).any()
    return features
