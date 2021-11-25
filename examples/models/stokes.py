from firedrake import *
from pyroteus_adjoint import *


class Parameters(object):
    viscosity = Constant(1.0)
    num_inputs = 24
    num_outputs = 3
    dofs_per_element = 3
    h_min = 1.0e-05
    h_max = 10.0

    def u_inflow(self, mesh):
        raise NotImplementedError

    def Re(self, fwd_sol):
        u = fwd_sol.split()[0]
        unorm = sqrt(dot(u, u))
        mesh = u.function_space().mesh()
        P0 = FunctionSpace(mesh, 'DG', 0)
        h = CellSize(mesh)
        return interpolate(0.5*h*unorm/self.viscosity, P0)


PETSc.Sys.popErrorHandler()
parameters = Parameters()
fields = ['up']


def get_function_spaces(mesh):
    return {'up': VectorFunctionSpace(mesh, 'CG', 2)*FunctionSpace(mesh, 'CG', 1)}


def get_solver(mesh_seq):

    def solver(i, ic):
        mesh = mesh_seq[i]
        W = mesh_seq.function_spaces['up'][i]

        # Extract test case specific parameters
        u_inflow = interpolate(parameters.u_inflow(mesh), W.sub(0))
        nu = parameters.viscosity

        # Boundary conditions
        inflow = DirichletBC(W.sub(0), u_inflow, 1)
        noslip = DirichletBC(W.sub(0), (0, 0), (3, 5))
        hole = DirichletBC(W.sub(0), 0, 4)
        bcs = [inflow, noslip, hole]

        # Forward solution
        up = Function(W, name='up_old')
        up.assign(ic['up'])

        # Variational form
        u, p = split(up)
        v, q = TestFunctions(W)
        F = nu*inner(grad(u), grad(v))*dx \
            - inner(p, div(v))*dx \
            - inner(q, div(u))*dx

        # Solve the Stokes equation
        sp = {
            'mat_type': 'aij',
            'snes_type': 'newtonls',
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_shift_type': 'inblocks',
        }
        solve(F == 0, up, bcs=bcs, solver_parameters=sp,
              options_prefix='forward', ad_block_tag='up')
        return {'up': up}

    return solver


def get_initial_condition(mesh_seq):
    fs = mesh_seq.function_spaces['up'][0]
    up = Function(fs)
    u, p = up.split()
    u.interpolate(parameters.u_inflow(mesh_seq[0]))
    return {'up': up}


def dwr_indicator(mesh_seq, up, up_star):
    u, p = up.split()
    v, q = up_star.split()
    nu = parameters.viscosity
    mesh_plus = u.function_space().mesh()
    P0_plus = FunctionSpace(mesh_plus, 'DG', 0)
    p0test = TestFunction(P0_plus)
    # n = FacetNormal(mesh_plus)

    # Evaluate dual-weighted residual
    F = p0test*nu*inner(grad(u), grad(v))*dx \
        - p0test*inner(p, div(v))*dx \
        - p0test*inner(q, div(u))*dx
    dwr_plus = assemble(F)
    # Psi = -p0test*div(nu*dot(grad(u), v))*dx \
    #     + p0test*inner(grad(p), v)*dx \
    #     - p0test*inner(q, div(u))*dx
    # Psi = assemble(Psi)
    # mass_term = p0test*TrialFunction(P0_plus)*dx
    # flux = p0test*(nu*dot(dot(grad(u), v), n) - p*dot(v, n))
    # flux_terms = (flux('+') + flux('-'))*dS
    # psi = Function(P0_plus)
    # sp = {
    #     'snes_type': 'ksponly',
    #     'ksp_type': 'preonly',
    #     'pc_type': 'jacobi',
    # }
    # solve(mass_term == flux_terms, psi, solver_parameters=sp)
    # dwr_plus = interpolate(abs(Psi + psi), P0_plus)

    # Project down to the base space
    P0 = FunctionSpace(mesh_seq[0], 'DG', 0)
    dwr = project(dwr_plus, P0)
    dwr.interpolate(abs(dwr))
    return dwr, dwr_plus
