from firedrake import *
from pyroteus_adjoint import *


def get_solutions(mesh, config, adjoint=True):
    """
    Solve forward and adjoint equations on a
    given mesh.

    This works only for steady-state problems.

    :arg mesh: input mesh
    :arg config: configuration file, which
        specifies the PDE and QoI
    :kwarg adjoint: solve the adjoint problem,
        as well as the forward problem?
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
    if adjoint:
        adj_kwargs = {"options_prefix": "adjoint"}
        sols = mesh_seq.solve_adjoint(adj_solver_kwargs=adj_kwargs)
        fwd_sols = sols[field].forward[0][0]
        adj_sols = sols[field].adjoint[0][0]
        return fwd_sols, adj_sols, mesh_seq
    else:
        sols = mesh_seq.get_checkpoints(run_final_subinterval=True)
        fwd_sols = sols[0][field]
        return fwd_sols, mesh_seq


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
                fs = FunctionSpace(V.mesh(), el.family(), el.degree())
                # ret[i] = [Function(fs).assign(fi.sub(j)) for j in range(Vi.shape[0])]
                ret[i] = [interpolate(fi[j], fs) for j in range(Vi.shape[0])]
        return ret
    elif len(V.shape) > 0:
        assert len(V.shape) == 1, "Tensor spaces not supported"
        el = V.ufl_element()
        fs = FunctionSpace(V.mesh(), el.family(), el.degree())
        # return {0: [Function(fs).assign(f.sub(i)) for i in range(V.shape[0])]}
        return {0: [interpolate(f[i], fs) for i in range(V.shape[0])]}
    else:
        return {0: [f]}


def split_into_components(f):
    """
    Extend the :attr:`split` method to apply
    to non-mixed :class:`Function`\s.
    """
    return [f] if f.function_space().value_size == 1 else f.split()


def indicate_errors(mesh, config, enrichment_method='h', retall=False):
    """
    Indicate errors according to the ``dwr_indicator``
    given in the configuration file.

    :arg mesh: input mesh
    :arg config: configuration file, which
        specifies the PDE and QoI
    :kwarg enrichment_method: how to enrich the
        finite element space?
    :kwarg retall: if ``True``, return the forward
        solution, adjoint solution, dual-weighted
        residual in enriched space, enriched
        adjoint solution and :class:`GoalOrientedMeshSeq`,
        in addition to the dual-weighted residual
        error indicator
    """
    fwd_sol, adj_sol, mesh_seq = get_solutions(mesh, config)

    # Solve PDE in enriched space
    adj_kwargs = {"options_prefix": "adjoint_enriched"}
    enriched_solutions = mesh_seq.global_enrichment(
        enrichment_method=enrichment_method,
        adj_solver_kwargs=adj_kwargs
    )
    field = config.fields[0]
    adj_sol_plus = enriched_solutions[field].adjoint[0][0]

    # Prolong
    V_plus = adj_sol_plus.function_space()
    fwd_sol_plg = Function(V_plus)
    prolong(fwd_sol, fwd_sol_plg)
    adj_sol_plg = Function(V_plus)
    prolong(adj_sol, adj_sol_plg)

    # Subtract prolonged adjoint solution from enriched version
    adj_error = Function(V_plus)
    adj_sols_plus = split_into_components(adj_sol_plus)
    adj_sols_plg = split_into_components(adj_sol_plg)
    for i, err in enumerate(split_into_components(adj_error)):
        err += adj_sols_plus[i] - adj_sols_plg[i]

    # Evaluate errors
    dwr, dwr_plus = config.dwr_indicator(mesh_seq, fwd_sol_plg, adj_error)
    if retall:
        return dwr, fwd_sol, adj_sol, dwr_plus, adj_error, mesh_seq
    else:
        return dwr
