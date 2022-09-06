# IMPORTANT INSTRUCTIONS: nRefs will be [1, ... , 25] but slicing will be [1, ... , 25][0 : 25]=[1, ... , 24]

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
                    help="nothing, _run1, _run_2, 300, 400, etc.",
                    type=str,
                    default="")
parser.add_argument("--nrefs_start", 
                    help="start (pythonic) slice to [1, 2, ... , 24, 25]",
                    type=int,
                    default="0")
parser.add_argument("--nrefs_end", 
                    help="end (pythonic) slice to [1, 2, ... , 24, 25]",
                    type=int,
                    default="25")
parser.add_argument("--spec_task", 
                    help="qoiError-nRefs, avgQoIError-acts",
                    type=str,
                    default="qoiError-nRefs")

args = parser.parse_args()
test_case = args.test_case
run_tag = args.run_tag
nrefs_start = args.nrefs_start
nrefs_end = args.nrefs_end
spec_task = args.spec_task

acts = ["Sigmoid", "Tanh", "ReLU"]
methods = ["GO"] + acts

qois = {}
qois["GO"] = np.load(f"steady_turbine/data/qois_GOanisotropic_{test_case}{run_tag}.npy")
qois["Sigmoid"] = np.load(f"steady_turbine/data/qois_MLanisotropic_{test_case}_1_F_64_100-4{run_tag}.npy")
qois["Tanh"] = np.load(f"steady_turbine/data/qois_MLanisotropic_{test_case}_1_F_64_100-4_Tanh{run_tag}.npy")
qois["ReLU"] = np.load(f"steady_turbine/data/qois_MLanisotropic_{test_case}_1_F_64_100-4_ReLU{run_tag}.npy")

conv = np.load(f"steady_turbine/data/qois_uniform_{test_case}.npy")[-1]

errors = {}
for method in methods:
    errors[method] = np.abs((qois[method] - conv) / conv)[nrefs_start : nrefs_end] # SLICED HERE

# get nrefs automatically from qois_GO
num_Refs = len(qois["GO"])
Refs = np.arange(1, num_Refs + 1)[nrefs_start : nrefs_end] # SLICED HERE

# choose saving directory
saving_dir = "steady_turbine/diffacts_qoiErrors-nRefs"
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

# qoiError-nRefs plot
if spec_task == "qoiError-nRefs":
    for method in methods:
        plt.plot(Refs, errors[method], label=method) # DATA ALREADY SLICED

    plt.xlabel("Number of refinements")
    plt.ylabel("QoI errors")
    plt.title(f"QoI Errors vs Number of Refinements on Different Activation Functions, {test_case}, {run_tag}")
    plt.legend()
    plt.savefig(f"{saving_dir}/qoiErrors-nRefs_{test_case}{run_tag}_{nrefs_start}-{nrefs_end}")

# avgQoIError-acts plot
if spec_task == "avgQoIError-acts":
    errors_avg = {}
    for method in methods:
        errors_avg[method] = np.mean(errors[method])

    plt.bar(errors_avg.keys(), errors_avg.values())
    plt.xlabel("Activation Funtions and GO")
    plt.ylabel(f"Average QoI errors of refinements from {nrefs_start} to {nrefs_end}")
    plt.title(f"Average QoI Errors of Refinements from {nrefs_start} to {nrefs_end} vs Different Activation Functions")
    plt.savefig(f"{saving_dir}/avgQoIErrors-methods_{test_case}{run_tag}_{nrefs_start}-{nrefs_end}")