import argparse
import importlib


# Parse for test case
parser = argparse.ArgumentParser()
parser.add_argument('model', help='The model')
parser.add_argument('test_case', help='The configuration file number')
parsed_args, unknown_args = parser.parse_known_args()
model = parsed_args.model
assert model in ['turbine']
test_case = int(parsed_args.test_case)
assert test_case in list(range(16))

# Load setup
setup = importlib.import_module(f'{model}.config')
setup.initialise(test_case)
meshgen = importlib.import_module(f'{model}.meshgen')

# Write geometry file
with open(f'{model}/meshes/{test_case}.geo', 'w+') as meshfile:
    meshfile.write(meshgen.generate_geo(setup))
