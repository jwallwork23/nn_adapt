"""
Run a given ``test_case`` of a ``model`` on a sequence of
uniformly refined meshes generated from the initial mesh.
"""
from nn_adapt.parse import Parser
from nn_adapt.solving import *
from thetis import print_output
import importlib
import numpy as np
from time import perf_counter


start_time = perf_counter()

# Parse user input
parser = Parser("run_uniform_refinement.py")
parser.parse_num_refinements(default=3)
parser.add_argument("--prolong", help="Use previous solution as initial guess", action="store_true")
parsed_args = parser.parse_args()
model = parsed_args.model
try:
    test_case = int(parsed_args.test_case)
    assert test_case > 0
except ValueError:
    test_case = parsed_args.test_case
num_refinements = parsed_args.num_refinements

# Setup
setup = importlib.import_module(f"{model}.config")
setup.initialise(test_case)
unit = setup.parameters.qoi_unit
mesh = Mesh(f"{model}/meshes/{test_case}.msh")
mh = MeshHierarchy(mesh, num_refinements)
tm = TransferManager()
kwargs = {}

# Run uniform refinement
qois, dofs, elements, times, niter = [], [], [], [], []
setup_time = perf_counter() - start_time
print_output(f"Test case {test_case}")
print_output(f"Setup time: {setup_time:.2f} seconds")
for i, mesh in enumerate(mh):
    start_time = perf_counter()
    print_output(f"  Mesh {i}")
    print_output(f"    Element count        = {mesh.num_cells()}")
    fwd_sol = get_solutions(mesh, setup, solve_adjoint=False, **kwargs)

    def prolong(V):
        """
        After the first iteration, prolong the previous
        solution as the initial guess.
        """
        ic = Function(V)
        tm.prolong(fwd_sol, ic)
        return ic

    if parsed_args.prolong:
        kwargs["init"] = prolong
    fs = fwd_sol.function_space()
    qoi = assemble(setup.get_qoi(mesh)(fwd_sol))
    time = perf_counter() - start_time
    print_output(f"    Quantity of Interest = {qoi} {unit}")
    print_output(f"    Runtime: {time:.2f} seconds")
    qois.append(qoi)
    dofs.append(sum(fs.dof_count))
    times.append(time)
    elements.append(mesh.num_cells())
    niter.append(1)
    np.save(f"{model}/data/qois_uniform_{test_case}", qois)
    np.save(f"{model}/data/dofs_uniform_{test_case}", dofs)
    np.save(f"{model}/data/elements_uniform_{test_case}", elements)
    np.save(f"{model}/data/times_uniform_{test_case}", times)
    np.save(f"{model}/data/niter_uniform_{test_case}", niter)
print_output(f"Setup time: {setup_time:.2f} seconds")
