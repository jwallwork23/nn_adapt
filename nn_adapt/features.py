"""
Functions for extracting feature data from configuration
files, meshes and solution fields.
"""
import firedrake
from firedrake.petsc import PETSc
from firedrake import op2
import numpy as np
from nn_adapt.solving import split_into_scalars
from pyroteus.metric import *
import ufl


__all__ = ["extract_features", "get_values_at_elements"]


@PETSc.Log.EventDecorator("Extract components")
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
    assert armin >= 1.0, f"An element has aspect ratio is less than one ({armin})"
    theta = firedrake.interpolate(ufl.atan(evecs[1, 1] / evecs[1, 0]), fs)
    h1 = firedrake.interpolate(ufl.cos(theta) ** 2 / ar + ufl.sin(theta) ** 2 * ar, fs)
    h2 = firedrake.interpolate((1 / ar - ar) * ufl.sin(theta) * ufl.cos(theta), fs)
    return density, h1, h2


@PETSc.Log.EventDecorator("Extract elementwise")
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
        assert fs.ufl_element().cell() == ufl.triangle, "Simplex meshes only"
    elif dim == 3:
        assert fs.ufl_element().cell() == ufl.tetrahedron, "Simplex meshes only"
    else:
        raise ValueError(f"Dimension {dim} not supported")
    el = fs.ufl_element()
    if el.sub_elements() == []:
        size = (dim + 1) * el.value_size() * el.degree()
    else:
        size = (dim + 1) * sum(
            sel.value_size() * sel.degree() for sel in el.sub_elements()
        )
    P0_vec = firedrake.VectorFunctionSpace(mesh, "DG", 0, dim=size)
    values = firedrake.Function(P0_vec)
    kernel = "for (int i=0; i < vertexwise.dofs; i++) elementwise[i] += vertexwise[i];"
    keys = {"vertexwise": (M, op2.READ), "elementwise": (values, op2.INC)}
    firedrake.par_loop(kernel, ufl.dx, keys)
    return values


@PETSc.Log.EventDecorator("Extract features")
def extract_features(config, fwd_sol, adj_sol, preproc="none"):
    """
    Extract features from the outputs of a run.

    :arg config: the configuration file
    :arg fwd_sol: the forward solution
    :arg adj_sol: the adjoint solution
    :kwarg preproc: preprocessor function
    """
    mesh = fwd_sol.function_space().mesh()

    # Features describing the mesh element
    with PETSc.Log.Event("Analyse element"):
        J = ufl.Jacobian(mesh)
        P0_ten = firedrake.TensorFunctionSpace(mesh, "DG", 0)
        JTJ = firedrake.interpolate(ufl.dot(ufl.transpose(J), J), P0_ten)
        d, h1, h2 = [p.dat.data for p in extract_components(JTJ)]

    # Features related to flow physics
    with PETSc.Log.Event("Extract physics"):
    #     drag = config.parameters.drag(mesh).dat.data
        ones = np.ones(len(d))
        nu = config.parameters.viscosity.values()[0] * ones  # NOTE: assumes constant
        b = config.parameters.depth * ones  # NOTE: assumes constant

    # Features describing the forward and adjoint solutions
    with PETSc.Log.Event("Extract DoFs"):
        fwd_sols = split_into_scalars(fwd_sol)
        adj_sols = split_into_scalars(adj_sol)
        sols = sum([fi for i, fi in fwd_sols.items()], start=[])
        sols += sum([ai for i, ai in adj_sols.items()], start=[])
        vals = [get_values_at_elements(s).dat.data for s in sols]

    # Combine the features together
    with PETSc.Log.Event("Combine features"):
        features = np.hstack(
            # (np.vstack([nu, drag, b, d, h1, h2]).transpose(), np.hstack(vals))
            (np.vstack([nu, b, d, h1, h2]).transpose(), np.hstack(vals))
        )
    assert not np.isnan(features).any()

    # Pre-process, if requested
    if preproc != "none":
        from nn_adapt.ann import preprocess_features

        features = preprocess_features(features, preproc=preproc)
    return features
