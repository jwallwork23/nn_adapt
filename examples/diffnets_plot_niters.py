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
                    help="Nothing, _run1, _run_2, etc.",
                    type=str,
                    default="")
args = parser.parse_args()
test_case = args.test_case
run_tag = args.run_tag

niter_GO = np.load("steady_turbine/data/niter_GOanisotropic_" + test_case + run_tag + ".npy")
niter_ML1 = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy")
niter_ML2 = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_2_F_64_100-4" + run_tag + ".npy")
niter_ML3 = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_3_F_64_100-4" + run_tag + ".npy")
niter_ML4 = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_4_F_64_100-4" + run_tag + ".npy")
niter_ML5 = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_5_F_64_100-4" + run_tag + ".npy")
niter_ML6 = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_6_F_64_100-4" + run_tag + ".npy")
niter_ML7 = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_7_F_64_100-4" + run_tag + ".npy")
niter_ML8 = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_8_F_64_100-4" + run_tag + ".npy")
niter_ML9 = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_9_F_64_100-4" + run_tag + ".npy")
niter_ML10 = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_10_F_64_100-4" + run_tag + ".npy")

n_refinements = np.arange(1, 26)
'''if test_case == "aligned_reversed":
    n_refinements = np.arange(1, 25)'''

# plot
plt.plot(n_refinements, niter_GO, label="GO")
plt.plot(n_refinements, niter_ML1, label="ML1")
plt.plot(n_refinements, niter_ML2, label="ML2")
plt.plot(n_refinements, niter_ML3, label="ML3")
plt.plot(n_refinements, niter_ML4, label="ML4")
plt.plot(n_refinements, niter_ML5, label="ML5")
plt.plot(n_refinements, niter_ML6, label="ML6")
plt.plot(n_refinements, niter_ML7, label="ML7")
plt.plot(n_refinements, niter_ML8, label="ML8")
plt.plot(n_refinements, niter_ML9, label="ML9")
plt.plot(n_refinements, niter_ML10, label="ML10")

plt.xlabel("Number of refinements")

plt.ylabel("Total Number of Iterations")

plt.title(f"Total Number of Iterations vs Number of refinements, {test_case}, {run_tag}")

plt.legend()

saving_dir = "steady_turbine/diffnets_plots_niters"
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

plt.savefig(f"{saving_dir}/niters_vs_n_refinements_{test_case}{run_tag}")