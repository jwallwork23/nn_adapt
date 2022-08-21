import string
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


dofs_go = np.load("steady_turbine/data/dofs_GOanisotropic_" + test_case + run_tag + ".npy")
dofs_ml1 = np.load("steady_turbine/data/dofs_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy")
dofs_ml1 = np.load("steady_turbine/data/dofs_MLanisotropic_" + test_case + "_2_F_64_100-4" + run_tag + ".npy")
dofs_ml3 = np.load("steady_turbine/data/dofs_MLanisotropic_" + test_case + "_3_F_64_100-4" + run_tag + ".npy")
dofs_ml1 = np.load("steady_turbine/data/dofs_MLanisotropic_" + test_case + "_4_F_64_100-4" + run_tag + ".npy")
dofs_ml5 = np.load("steady_turbine/data/dofs_MLanisotropic_" + test_case + "_5_F_64_100-4" + run_tag + ".npy")
dofs_ml1 = np.load("steady_turbine/data/dofs_MLanisotropic_" + test_case + "_6_F_64_100-4" + run_tag + ".npy")
dofs_ml7 = np.load("steady_turbine/data/dofs_MLanisotropic_" + test_case + "_7_F_64_100-4" + run_tag + ".npy")
dofs_ml1 = np.load("steady_turbine/data/dofs_MLanisotropic_" + test_case + "_8_F_64_100-4" + run_tag + ".npy")
dofs_ml1 = np.load("steady_turbine/data/dofs_MLanisotropic_" + test_case + "_9_F_64_100-4" + run_tag + ".npy")
dofs_ml10 = np.load("steady_turbine/data/dofs_MLanisotropic_" + test_case + "_10_F_64_100-4" + run_tag + ".npy")

n_refinements = np.arange(1, 26)

plt.plot(n_refinements, dofs_go, label="GO")
plt.plot(n_refinements, dofs_ml1, label="ML1")
plt.plot(n_refinements, dofs_ml1, label="ML2")
plt.plot(n_refinements, dofs_ml3, label="ML3")
plt.plot(n_refinements, dofs_ml1, label="ML4")
plt.plot(n_refinements, dofs_ml5, label="ML5")
plt.plot(n_refinements, dofs_ml1, label="ML6")
plt.plot(n_refinements, dofs_ml7, label="ML7")
plt.plot(n_refinements, dofs_ml1, label="ML8")
plt.plot(n_refinements, dofs_ml1, label="ML9")
plt.plot(n_refinements, dofs_ml10, label="ML10")

plt.xlabel("Number of refinements")

plt.ylabel("DoFs")

plt.title(f"DoF Count vs Number of Refinements, {test_case}, {run_tag}")

plt.legend()

saving_dir = "steady_turbine/diffnets_plots_dofs"
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

plt.savefig(f"{saving_dir}/dofs_vs_n_refinements_{test_case}{run_tag}")
