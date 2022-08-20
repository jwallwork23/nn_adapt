from firedrake import *
from pyroteus_adjoint import *
from firedrake.petsc import PETSc
import nn_adapt.model

'''
A memory hungry method solving time dependent PDE.
'''
class Parameters(nn_adapt.model.Parameters):
    """
    Class encapsulating all parameters required for a simple
    Burgers equation test case.
    """

    qoi_name = "right boundary integral"
    qoi_unit = r"m\,s^{-1}"

    # Adaptation parameters
    h_min = 1.0e-10  # Minimum metric magnitude
    h_max = 1.0  # Maximum metric magnitude

    # Physical parameters
    viscosity_coefficient = 0.0001
    initial_speed = 1.0

    # Timestepping parameters
    timestep = 0.05

    solver_parameters = {}
    adjoint_solver_parameters = {}

    def bathymetry(self, mesh):
        """
        Compute the bathymetry field on the current `mesh`.
        Note that there isn't really a concept of bathymetry
        for Burgers equation. It is kept constant and should
        be ignored by the network.
        """
        P0_2d = FunctionSpace(mesh, "DG", 0)
        return Function(P0_2d).assign(1.0)

    def drag(self, mesh):
        """
        Compute the bathymetry field on the current `mesh`.
        Note that there isn't really a concept of bathymetry
        for Burgers equation. It is kept constant and should
        be ignored by the network.
        """
        P0_2d = FunctionSpace(mesh, "DG", 0)
        return Function(P0_2d).assign(1.0)

    def viscosity(self, mesh):
        """
        Compute the viscosity coefficient on the current `mesh`.
        """
        P0_2d = FunctionSpace(mesh, "DG", 0)
        return Function(P0_2d).assign(self.viscosity_coefficient)

    def ic(self, mesh):
        """
        Initial condition
        """
        x, y = SpatialCoordinate(mesh)
        expr = self.initial_speed * sin(pi * x)
        return as_vector([expr, 0])
    
    
def get_function_spaces(mesh):
    return {"u": VectorFunctionSpace(mesh, "CG", 2)}


def get_form(mesh_seq):
    def form(index, solutions):
        u, u_ = solutions["u"]
        P = mesh_seq.time_partition
        dt = Constant(P.timesteps[index])

        # Specify viscosity coefficient
        nu = Constant(0.0001)

        # Setup variational problem
        v = TestFunction(u.function_space())
        F = (
            inner((u - u_) / dt, v) * dx
            + inner(dot(u, nabla_grad(u)), v) * dx
            + nu * inner(grad(u), grad(v)) * dx
        )
        return F

    return form


def get_solver(mesh_seq):
    def solver(index, ic):
        function_space = mesh_seq.function_spaces["u"][index]
        u = Function(function_space)

        # Initialise 'lagged' solution
        u_ = Function(function_space, name="u_old")
        u_.assign(ic["u"])

        # Define form
        F = mesh_seq.form(index, {"u": (u, u_)})

        # Time integrate from t_start to t_end
        P = mesh_seq.time_partition
        t_start, t_end = P.subintervals[index]
        dt = P.timesteps[index]
        t = t_start
        step = 0
        while t < t_end - 1.0e-05:
            step += 1
            print(step)
            solve(F == 0, u, ad_block_tag="u")
            u_.assign(u)
            t += dt
        return {"u": u}

    return solver


def get_initial_condition(mesh_seq):
    fs = mesh_seq.function_spaces["u"][0]
    x, y = SpatialCoordinate(mesh_seq[0])
    return {"u": interpolate(as_vector([sin(pi * x), 0]), fs)}

def get_qoi(mesh_seq, solutions, index):
    def end_time_qoi():
        u = solutions["u"]
        return inner(u, u) * ds(2)

    def time_integrated_qoi(t):
        dt = Constant(mesh_seq.time_partition[index].timestep)
        u = solutions["u"]
        return dt * inner(u, u) * ds(2)
    
    if mesh_seq.qoi_type == "end_time":
        return end_time_qoi
    else:
        return time_integrated_qoi


PETSc.Sys.popErrorHandler()
parameters = Parameters()

class pyroteus_burgers():
    
    def __init__(self, meshes, ic, **kwargs):
        
        self.meshes = meshes
        self.kwargs = kwargs
        try:
            self.nu = [parameters.viscosity(mesh) for mesh in meshes]
            self.num_subintervals = len(meshes)
        except:
            self.nu = parameters.viscosity(meshes)
            self.num_subintervals = 1
    
    def setups(self):
        
        fields = ["u"]
        
        dt = 0.1
        steps_subintervals = 3
        end_time = self.num_subintervals * steps_subintervals * dt
        
        timesteps_per_export = 1
        
        time_partition = TimePartition(
            end_time,
            self.num_subintervals,
            dt,
            fields,
            timesteps_per_export=timesteps_per_export,
        )
        
        self._mesh_seq = GoalOrientedMeshSeq(
            time_partition,
            self.meshes,
            get_function_spaces=get_function_spaces,
            get_initial_condition=get_initial_condition,
            get_form=get_form,
            get_solver=get_solver,
            get_qoi=get_qoi,
            qoi_type="end_time",
        )
        
       
    def iterate(self):
        self.setups()
        self._solutions, self._indicators = self._mesh_seq.indicate_errors(
            enrichment_kwargs={"enrichment_method": "h"}
        )
    
    
    def integrate(self, item):
        result = [0 for _ in range(self._mesh_seq.num_subintervals)]
        steps = self._mesh_seq.time_partition.timesteps_per_subinterval
        
        for id, list in  enumerate(item):
            for element in list:
                result[id] += element
            result[id] = product((result[id], 1/steps[id]))
            
        return result
        
    @property
    def fwd_sol(self):
        return self.integrate(self._solutions["u"]["forward"])
    
    @property
    def adj_sol(self):
        return self.integrate(self._solutions["u"]["adjoint"])
    
    @property
    def qoi(self):
        return self._mesh_seq.J
    
    @property
    def indicators(self):
        return self.integrate(self._indicators)
    
    
mesh = [UnitSquareMesh(15, 15), UnitSquareMesh(12, 17)]
ic = 0
demo = Solver_n4one(mesh, ic)

demo.iterate()

        