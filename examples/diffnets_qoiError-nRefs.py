# IMPORTANT INSTRUCTIONS: nRefs here will use python array-like indexing

from runpy import run_path
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


parser = argparse.ArgumentParser(description="Specify test case and run tag.")
parser.add_argument("--test_case", 
                    help="aligned, offset, aligned_reversed or trench",
                    type=str,
                    default="aligned")
parser.add_argument("--run_tag", 
                    help="nothing, _run1, _run_2, etc.",
                    type=str,
                    default="")
'''parser.add_argument("--nrefs_start", 
                    help="Choose from 1 to 25, incl.",
                    type=int,
                    default="20")
parser.add_argument("--nrefs_end", 
                    help="Choose from 1 to 25, also incl.",
                    type=int,
                    default="25")'''

args = parser.parse_args()
test_case = args.test_case
run_tag = args.run_tag
'''nrefs_start = args.nrefs_start
nrefs_end = args.nrefs_end'''

keys = {"ML1", "ML2", "ML3", "ML4", "ML5", "ML6", "ML7", "ML8", "ML9", "ML10"}

qois = {}
qois["GO"] = np.load(f"steady_turbine/data/qois_GOanisotropic_{test_case}{run_tag}.npy")
for key in keys:
    qois[key] = np.load(f"steady_turbine/data/qois_MLanisotropic_{test_case}_" + key[2:] + f"_F_64_100-4{run_tag}.npy")

conv = np.load(f"steady_turbine/data/qois_uniform_{test_case}.npy")[-1]

errors = {}
errors["GO"] = np.abs((qois["GO"] - conv) / conv)
for key in keys:
    errors[key] = np.abs((qois[key] - conv) / conv)

# get nrefs automatically from qois_GO
num_Refs = len(qois["GO"])
Refs = np.arange(num_Refs)

# plot
for key in keys:
    plt.plot(Refs, errors[key], label=key)

plt.xlabel("Number of refinements")

plt.ylabel("QoI errors")

plt.title(f"QoI Errors vs Number of Refinements, {test_case}, {run_tag}")

plt.legend()

saving_dir = "steady_turbine/diffnets_qoiErrors-nRefs"
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

plt.savefig(f"{saving_dir}/qoiErrors-nRefs_{test_case}{run_tag}")
