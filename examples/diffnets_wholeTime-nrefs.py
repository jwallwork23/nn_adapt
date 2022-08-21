from runpy import run_path
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


parser = argparse.ArgumentParser(description="Specify test case and run tag.")
parser.add_argument("--test_case", 
                    help="Aligned, offset, aligned_reversed or trench",
                    type=str,
                    default="aligned")
parser.add_argument("--run_tag", 
                    help="_, _run1, _run_2, etc.",
                    type=str,
                    default="")
parser.add_argument("--nrefs_start", 
                    help="Choose from 1 to 25, incl.",
                    type=int,
                    default="20")
parser.add_argument("--nrefs_end", 
                    help="Choose from 1 to 25, also incl.",
                    type=int,
                    default="25")


args = parser.parse_args()
test_case = args.test_case
run_tag = args.run_tag
nrefs_start = args.nrefs_start
nrefs_end = args.nrefs_end

# parts
parts = ["forward", "adjoint", "estimator", "metric", "adapt"]

keys = ["all"] + parts

# times_all and times_part on 20 ~ 25th refinements
times_GO = {}
times_ML1 = {}
times_ML2 = {}
times_ML3 = {}
times_ML4 = {}
times_ML5 = {}
times_ML6 = {}
times_ML7 = {}
times_ML8 = {}
times_ML9 = {}
times_ML10 = {}

times_GO["all"] = np.load("steady_turbine/data/times_all_GOanisotropic_" + test_case + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]
times_ML1["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]
times_ML2["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_2_F_64_100-4" + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]
times_ML3["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_3_F_64_100-4" + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]
times_ML4["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_4_F_64_100-4" + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]
times_ML5["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_5_F_64_100-4" + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]
times_ML6["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_6_F_64_100-4" + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]
times_ML7["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_7_F_64_100-4" + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]
times_ML8["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_8_F_64_100-4" + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]
times_ML9["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_9_F_64_100-4" + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]
times_ML10["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_10_F_64_100-4" + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]

for part in parts:
    times_GO[part] = np.load("steady_turbine/data/times_" + part + "_GOanisotropic_" + test_case + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]
    times_ML1[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]
    times_ML2[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_2_F_64_100-4" + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]
    times_ML3[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_3_F_64_100-4" + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]
    times_ML4[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_4_F_64_100-4" + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]
    times_ML5[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_5_F_64_100-4" + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]
    times_ML6[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_6_F_64_100-4" + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]
    times_ML7[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_7_F_64_100-4" + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]
    times_ML8[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_8_F_64_100-4" + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]
    times_ML9[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_9_F_64_100-4" + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]
    times_ML10[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_10_F_64_100-4" + run_tag + ".npy")[nrefs_start - 1 : nrefs_end]

# times_all_avg and times_part_avg (averaged over 20 ~ 25th refinements)
times_GO_avg = {}
times_ML1_avg = {}
times_ML2_avg = {}
times_ML3_avg = {}
times_ML4_avg = {}
times_ML5_avg = {}
times_ML6_avg = {}
times_ML7_avg = {}
times_ML8_avg = {}
times_ML9_avg = {}
times_ML10_avg = {}

for key in keys:
    times_GO_avg[key] = np.mean(times_GO[key])
    times_ML1_avg[key] = np.mean(times_ML1[key])
    times_ML2_avg[key] = np.mean(times_ML2[key])
    times_ML3_avg[key] = np.mean(times_ML3[key])
    times_ML4_avg[key] = np.mean(times_ML4[key])
    times_ML5_avg[key] = np.mean(times_ML5[key])
    times_ML6_avg[key] = np.mean(times_ML6[key])
    times_ML7_avg[key] = np.mean(times_ML7[key])
    times_ML8_avg[key] = np.mean(times_ML8[key])
    times_ML9_avg[key] = np.mean(times_ML9[key])
    times_ML10_avg[key] = np.mean(times_ML10[key])

# taylor nrefs
n_refinements = np.arange(nrefs_start, nrefs_end + 1)

# they are a bit different
#print(np.allclose(times_GO["all"], times_GO["forward"] + times_GO["adjoint"] + times_GO["estimator"] + times_GO["metric"] + times_GO["adapt"]))
#print(times_GO["all"])
#print(times_GO["forward"] + times_GO["adjoint"] + times_GO["estimator"] + times_GO["metric"] + times_GO["adapt"])

plt.plot(n_refinements, times_GO["all"], label="GO")
plt.plot(n_refinements, times_ML1["all"], label="ML1")
plt.plot(n_refinements, times_ML2["all"], label="ML2")
plt.plot(n_refinements, times_ML3["all"], label="ML3")
plt.plot(n_refinements, times_ML4["all"], label="ML4")
plt.plot(n_refinements, times_ML5["all"], label="ML5")
plt.plot(n_refinements, times_ML6["all"], label="ML6")
plt.plot(n_refinements, times_ML7["all"], label="ML7")
plt.plot(n_refinements, times_ML8["all"], label="ML8")
plt.plot(n_refinements, times_ML9["all"], label="ML9")
plt.plot(n_refinements, times_ML10["all"], label="ML10")

plt.xlabel(f"Number of Refinements")

plt.ylabel("Whole Times")

plt.title(f"Whole Times vs {nrefs_start}th to {nrefs_end}th Refinements")

plt.legend()

saving_dir = "steady_turbine/diffnets_wholeTimes-nRefs"
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

plt.savefig(f"{saving_dir}/wholeTimes-nRefs_{test_case}_{nrefs_start}-{nrefs_end}{run_tag}")

