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
parser.add_argument("--nrefs_start", 
                    help="start (pythonic) slice to [1, 2, ... , 22, 23, 24, 25]",
                    type=int,
                    default="20")
parser.add_argument("--nrefs_end", 
                    help="end (pythonic) slice to [1, 2, ... , 22, 23, 24, 25]",
                    type=int,
                    default="25")
parser.add_argument("--spec_task", 
                    help="qoiError-nRefs, avgQoIError-methods",
                    type=str,
                    default="qoiError-nRefs")

args = parser.parse_args()
test_case = args.test_case
run_tag = args.run_tag
nrefs_start = args.nrefs_start
nrefs_end = args.nrefs_end
spec_task = args.spec_task

MLs = ["ML1", "ML2", "ML3", "ML4", "ML5", "ML6", "ML7", "ML8", "ML9", "ML10"]
methods = ["GO"] + MLs

qois = {}
qois["GO"] = np.load(f"steady_turbine/data/qois_GOanisotropic_{test_case}{run_tag}.npy")
for MLi in MLs:
    qois[MLi] = np.load(f"steady_turbine/data/qois_MLanisotropic_{test_case}_" + MLi[2:] + f"_F_64_100-4{run_tag}.npy")

conv = np.load(f"steady_turbine/data/qois_uniform_{test_case}.npy")[-1]

errors = {}
for method in methods:
    errors[method] = np.abs((qois[method] - conv) / conv)[nrefs_start : nrefs_end] # SLICED HERE

# get nrefs automatically from qois_GO
num_Refs = len(qois["GO"])
Refs = np.arange(1, num_Refs + 1)[nrefs_start : nrefs_end] # SLICED HERE

saving_dir = "steady_turbine/diffnets_qoiErrors-nRefs"
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

if spec_task == "qoiError-nRefs":
    # plot
    for method in methods:
        plt.plot(Refs, errors[method], label=method) # DATA ALREADY SLICED
    
    plt.xlabel("Number of refinements")
    plt.ylabel("QoI errors")
    plt.title(f"QoI Errors vs Number of Refinements, {test_case}, {run_tag}")
    plt.legend()

    plt.savefig(f"{saving_dir}/qoiErrors-nRefs_{test_case}{run_tag}_{nrefs_start}-{nrefs_end}")
    plt.close()

if spec_task == "avgQoIError-methods":
    errors_avg = {}
    for method in methods:
        errors_avg[method] = np.mean(errors[method])

    plt.bar(errors_avg.keys(), errors_avg.values())
    plt.xlabel("Methods")
    plt.ylabel(f"Average QoI errors of refinements from {nrefs_start} to {nrefs_end}")
    plt.title(f"Average QoI Errors of Refinements from {nrefs_start} to {nrefs_end} vs Methods")
    plt.savefig(f"{saving_dir}/avgQoIErrors-methods_{test_case}{run_tag}_{nrefs_start}-{nrefs_end}")
    plt.close()