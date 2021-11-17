from pyroteus_adjoint import *


def get_solutions(mesh, config):
    """
    Solve forward and adjoint equations on a
    given mesh.

    This works only for steady-state problems.

    :arg mesh: input mesh
    :arg config: configuration file, which
        specifies the PDE and QoI
    :return: forward solution, adjoint solution
        and :class:`GoalOrientedMeshSeq`
    """
    fields = config.fields
    assert len(fields) == 1, "Multiple fields not supported"
    field = fields[0]

    # We restrict attention to steady-state problems
    dt = 20.0
    end_time = dt
    num_subintervals = 1
    time_partition = TimePartition(
        end_time, num_subintervals, dt, fields, debug=True,
    )

    # Create MeshSeq object
    mesh_seq = GoalOrientedMeshSeq(
        time_partition, mesh, config.get_function_spaces,
        config.get_initial_condition, config.get_solver,
        config.get_qoi, qoi_type="end_time", steady=True,
    )

    # Solve forward and adjoint problems
    adj_kwargs = {"options_prefix": "adjoint"}
    sols = mesh_seq.solve_adjoint(adj_solver_kwargs=adj_kwargs)
    fwd_sols = sols[field].forward[0][0]
    adj_sols = sols[field].adjoint[0][0]
    return fwd_sols, adj_sols, mesh_seq


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
                ret[i] = [fi.sub(j) for j in range(Vi.shape[0])]
        return ret
    elif len(V.shape) > 0:
        assert len(V.shape) == 1, "Tensor spaces not supported"
        return {0: [f.sub(i)] for i in range(V.shape[0])}
    else:
        return {0: [f]}
