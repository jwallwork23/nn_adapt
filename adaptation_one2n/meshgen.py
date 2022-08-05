"""
Generate the mesh for configuration ``case``
of a given ``model``.
"""
import argparse
import importlib
import sys


# Parse for test case
parser = argparse.ArgumentParser(prog="meshgen.py")
parser.add_argument("model", help="The model")
parser.add_argument("case", help="The configuration file name")
parsed_args, unknown_args = parser.parse_known_args()
model = parsed_args.model
reverse = False
try:
    case = int(parsed_args.case)
    assert case > 0
except ValueError:
    case = parsed_args.case
    reverse = "reversed" in case

# Load setup
setup = importlib.import_module(f"{model}.config")
setup.initialise(case)
meshgen = importlib.import_module(f"{model}.meshgen")

# Write geometry file
code = meshgen.generate_geo(setup, reverse=reverse)
if code is None:
    sys.exit(0)
with open(f"{model}/meshes/{case}.geo", "w+") as meshfile:
    meshfile.write(code)
