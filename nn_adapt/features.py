"""
Functions for extracting feature data from configuration
files, meshes and solution fields.
"""
import firedrake
from firedrake.petsc import PETSc
from firedrake import op2
import numpy as np
from pyroteus.metric import *
import ufl
from nn_adapt.solving import dwr_indicator


__all__ = ["extract_features", "get_values_at_elements", "collect_features"]


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
def get_values_at_elements(f):
    """
    Extract the values for all degrees of freedom associated
    with each element.

    :arg f: some :class:`Function`
    :return: a vector :class:`Function` holding all DoFs of `f`
    """
    fs = f.function_space()
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
        p = el.degree()
        size = el.value_size() * (p + 1) * (p + 2) // 2
    else:
        size = 0
        for sel in el.sub_elements():
            p = sel.degree()
            size += sel.value_size() * (p + 1) * (p + 2) // 2
    P0_vec = firedrake.VectorFunctionSpace(mesh, "DG", 0, dim=size)
    values = firedrake.Function(P0_vec)
    kernel = "for (int i=0; i < vertexwise.dofs; i++) elementwise[i] += vertexwise[i];"
    keys = {"vertexwise": (f, op2.READ), "elementwise": (values, op2.INC)}
    firedrake.par_loop(kernel, ufl.dx, keys)
    return values


@PETSc.Log.EventDecorator("Extract at centroids")
def get_values_at_centroids(f):
    """
    Extract the values for the function at each element centroid,
    along with all derivatives up to the :math:`p^{th}`, where
    :math:`p` is the polynomial degree.

    :arg f: some :class:`Function`
    :return: a vector :class:`Function` holding all DoFs of `f`
    """
    fs = f.function_space()
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
        p = el.degree()
        degrees = [p]
        size = el.value_size() * (p + 1) * (p + 2) // 2
        funcs = [f]
    else:
        size = 0
        degrees = [sel.degree() for sel in el.sub_elements()]
        for sel, p in zip(el.sub_elements(), degrees):
            size += sel.value_size() * (p + 1) * (p + 2) // 2
        funcs = f
    values = firedrake.Function(firedrake.VectorFunctionSpace(mesh, "DG", 0, dim=size))
    P0 = firedrake.FunctionSpace(mesh, "DG", 0)
    P0_vec = firedrake.VectorFunctionSpace(mesh, "DG", 0)
    P0_ten = firedrake.TensorFunctionSpace(mesh, "DG", 0)
    i = 0
    for func, p in zip(funcs, degrees):
        values.dat.data[:, i] = firedrake.project(func, P0).dat.data_ro
        i += 1
        if p == 0:
            continue
        g = firedrake.project(ufl.grad(func), P0_vec)
        values.dat.data[:, i] = g.dat.data_ro[:, 0]
        values.dat.data[:, i + 1] = g.dat.data_ro[:, 1]
        i += 2
        if p == 1:
            continue
        H = firedrake.project(ufl.grad(ufl.grad(func)), P0_ten)
        values.dat.data[:, i] = H.dat.data_ro[:, 0, 0]
        values.dat.data[:, i + 1] = 0.5 * (
            H.dat.data_ro[:, 0, 1] + H.dat.data_ro[:, 1, 0]
        )
        values.dat.data[:, i + 2] = H.dat.data_ro[:, 1, 1]
        i += 3
        if p > 2:
            raise NotImplementedError(
                "Polynomial degrees greater than 2 not yet considered"
            )
    return values


def split_into_scalars(f):
    """
    Given a :class:`Function`, split it into
    components from its constituent scalar
    spaces.

    If it is not mixed then no splitting is
    required.

    :arg f: the mixed :class:`Function`
    :return: a dictionary containing the
        nested structure of the mixed function
    """
    V = f.function_space()
    if V.value_size > 1:
        subspaces = [V.sub(i) for i in range(len(V.node_count))]
        ret = {}
        for i, (Vi, fi) in enumerate(zip(subspaces, f.split())):
            if len(Vi.shape) == 0:
                ret[i] = [fi]
            else:
                assert len(Vi.shape) == 1, "Tensor spaces not supported"
                el = Vi.ufl_element()
                fs = firedrake.FunctionSpace(V.mesh(), el.family(), el.degree())
                ret[i] = [firedrake.interpolate(fi[j], fs) for j in range(Vi.shape[0])]
        return ret
    elif len(V.shape) > 0:
        assert len(V.shape) == 1, "Tensor spaces not supported"
        el = V.ufl_element()
        fs = firedrake.FunctionSpace(V.mesh(), el.family(), el.degree())
        return {0: [firedrake.interpolate(f[i], fs) for i in range(V.shape[0])]}
    else:
        return {0: [f]}


def extract_array(f, mesh=None, centroid=False, project=False):
    r"""
    Extract a cell-wise data array from a :class:`Constant` or
    :class:`Function`.

    For constants and scalar fields, this will be an :math:`n\times 1`
    array, where :math:`n` is the number of mesh elements. For a mixed
    field with :math:`m` components, it will be :math:`n\times m`.

    :arg f: the :class:`Constant` or :class:`Function`
    :kwarg mesh: the underlying :class:`MeshGeometry`
    :kwarg project: if ``True``, project the field into
        :math:`\mathbb P0` space
    """
    mesh = mesh or f.ufl_domain()
    if isinstance(f, firedrake.Constant):
        ones = np.ones(mesh.num_cells())
        assert len(f.values()) == 1
        return f.values()[0] * ones
    elif not isinstance(f, firedrake.Function):
        raise ValueError(f"Unexpected input type {type(f)}")
    if project:
        if len(f.function_space().shape) > 0:
            raise NotImplementedError("Can currently only project scalar fields")  # TODO
        element = f.ufl_element()
        if (element.family(), element.degree()) != ("Discontinuous Lagrange", 0):
            P0 = FunctionSpace(mesh, "DG", 0)
            f = project(f, P0)
    s = sum([fi for i, fi in split_into_scalars(f).items()], start=[])
    get = get_values_at_centroids if centroid else get_values_at_elements
    if len(s) == 1:
        return get(s[0]).dat.data
    else:
        return np.hstack([get(si).dat.data for si in s])


@PETSc.Log.EventDecorator("Extract features")
def extract_features(config, fwd_sol, adj_sol, preproc="none"):
    """
    Extract features from the outputs of a run.

    :arg config: the configuration file
    :arg fwd_sol: the forward solution
    :arg adj_sol: the adjoint solution
    :kwarg preproc: preprocessor function
    :return: a list of feature arrays
    """
    mesh = fwd_sol.function_space().mesh()

    # Coarse-grained DWR estimator
    with PETSc.Log.Event("Extract estimator"):
        dwr = dwr_indicator(config, mesh, fwd_sol, adj_sol)

    # Features describing the mesh element
    with PETSc.Log.Event("Analyse element"):
        P0_ten = firedrake.TensorFunctionSpace(mesh, "DG", 0)

        # Element size, orientation and shape
        J = ufl.Jacobian(mesh)
        JTJ = firedrake.interpolate(ufl.dot(ufl.transpose(J), J), P0_ten)
        d, h1, h2 = (extract_array(p) for p in extract_components(JTJ))

        # Is the element on the boundary?
        p0test = firedrake.TestFunction(dwr.function_space())
        bnd = firedrake.assemble(p0test * ufl.ds).dat.data

    # Combine the features together
    features = {
        "estimator_coarse": extract_array(dwr),
        "physics_drag": extract_array(config.parameters.drag(mesh)),
        "physics_viscosity": extract_array(config.parameters.viscosity(mesh), project=True),
        "physics_bathymetry": extract_array(config.parameters.bathymetry(mesh), project=True),
        "mesh_d": d,
        "mesh_h1": h1,
        "mesh_h2": h2,
        "mesh_bnd": bnd,
        "forward_dofs": extract_array(fwd_sol, centroid=True),
        "adjoint_dofs": extract_array(adj_sol, centroid=True),
    }
    for key, value in features.items():
        assert not np.isnan(value).any()

    # Pre-process, if requested
    if preproc != "none":
        from nn_adapt.ann import preprocess_features

        features = preprocess_features(features, preproc=preproc)
    return features


def collect_features(feature_dict, preproc="none"):
    """
    Given a dictionary of feature arrays, stack their
    data appropriately to be fed into a neural network.

    :arg feature_dict: dictionary containing feature data
    :kwarg preproc: preprocessor function
    """

    # Pre-process, if requested
    if preproc != "none":
        from nn_adapt.ann import preprocess_features

        feature_dict = preprocess_features(feature_dict, preproc=preproc)

    # Stack appropriately
    dofs = [feature for key, feature in feature_dict.items() if "dofs" in key]
    nodofs = [feature for key, feature in feature_dict.items() if "dofs" not in key]
    return np.hstack((np.vstack(nodofs).transpose(), np.hstack(dofs)))
