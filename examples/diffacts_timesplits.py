import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# only for selecting the scope of refinements for averaging
# default setting is averaging from 20th to 25th refinements
nRefs = len(np.load("steady_turbine/data/times_all_MLanisotropic_aligned_1_F_64_100-4.npy"))

parser = argparse.ArgumentParser(description="Specify test case and run tag.")
parser.add_argument("--test_case", 
                    help="Aligned, offset, aligned_reversed or trench",
                    type=str,
                    default="aligned")
parser.add_argument("--run_tag", 
                    help="Nothing, _run1, _run_2, etc.",
                    type=str,
                    default="")
parser.add_argument("--slice_a", 
                    help="lower bound (incl.) refinement a in Python indexing [a, b)",
                    type=int,
                    default=20)
parser.add_argument("--slice_b", 
                    help="upper bound (excl.) refinement b in Python indexing [a, b)",
                    type=int,
                    default=nRefs)
args = parser.parse_args()
test_case = args.test_case
run_tag = args.run_tag
slice_a = args.slice_a
slice_b = args.slice_b

# parts
parts = ["forward", "adjoint", "estimator", "metric", "adapt"]
keys = ["all"] + parts

# times_all and times_part on specified refinements
#times_GO = {}
times_sigm = {}
times_relu = {}
times_tanh = {}

#times_GO["all"] = np.load("steady_turbine/data/times_all_GOanisotropic_" + test_case + run_tag + ".npy")[slice_a : slice_b]
times_sigm["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy")[slice_a : slice_b]
times_tanh["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4_Tanh" + run_tag + ".npy")[slice_a : slice_b]
times_relu["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4_ReLU" + run_tag + ".npy")[slice_a : slice_b]

for part in parts:
    #times_GO[part] = np.load("steady_turbine/data/times_" + part + "_GOanisotropic_" + test_case + run_tag + ".npy")[slice_a : slice_b]
    times_sigm[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy")[slice_a : slice_b]
    times_tanh[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_1_F_64_100-4_Tanh" + run_tag + ".npy")[slice_a : slice_b]
    times_relu[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_1_F_64_100-4_ReLU" + run_tag + ".npy")[slice_a : slice_b]
    

# times_all_avg and times_part_avg (averaged over 20 ~ 25th refinements)
#times_GO_avg = {}
times_sigm_avg = {}
times_tanh_avg = {}
times_relu_avg = {}

for key in keys:
    #times_GO_avg[key] = np.mean(times_GO[key])
    times_sigm_avg[key] = np.mean(times_sigm[key])
    times_tanh_avg[key] = np.mean(times_tanh[key])
    times_relu_avg[key] = np.mean(times_relu[key])

# stacked bars of time splits (avraged over 20~25th refinements)
methods = ["Sigmoid", "ReLU", "Tanh"]

mat = np.stack((#list(times_GO_avg.values()),
                list(times_sigm_avg.values()),
                list(times_tanh_avg.values()),
                list(times_relu_avg.values()),
                ))

all = mat[:,0]
forward = mat[:,1]
adjoint = mat[:,2]
estimator = mat[:,3]
metric = mat[:,4]
adapt = mat[:,5]


stk1 = plt.bar(methods, forward, 0.5, label="forward")
stk2 = plt.bar(methods, adjoint, 0.5, label="adjoint", bottom=forward)
stk3 = plt.bar(methods, estimator, 0.5, label="estimator", bottom=forward+adjoint)
stk4 = plt.bar(methods, metric, 0.5, label="metric", bottom=forward+adjoint+estimator)
stk5 = plt.bar(methods, adapt, 0.5, label="adapt", bottom=forward+adjoint+estimator+metric)

plt.legend()

plt.title("Time Splits of Different Activation Functions of " + test_case + run_tag)

for s1, s2, s3, s4, s5 in zip(stk1, stk2, stk3, stk4, stk5):
    h1 = s1.get_height()
    plt.text(s1.get_x() + s1.get_width() / 2., h1 / 2., "%.2f" % h1, ha="center")
    
    h2 = s2.get_height()
    plt.text(s2.get_x() + s2.get_width() / 2., h1 + h2 / 2., "%.2f" % h2, ha="center")

    h3 = s3.get_height()
    plt.text(s3.get_x() + s3.get_width() / 2., h1 + h2 + h3 / 2., "%.2f" % h3, ha="center")

    h4 = s4.get_height()
    plt.text(s4.get_x() + s4.get_width() / 2., h1 + h2 + h3 + h4 / 2., "%.2f" % h4, ha="center")

    h5 = s5.get_height()
    plt.text(s5.get_x() + s5.get_width() / 2., h1 + h2 + h3 + h4 + h5 / 2., "%.2f" % h5, ha="center")

    plt.text(s5.get_x() + s5.get_width() / 2., h1 + h2 + h3 + h4 + h5 * 2., "%.2f" % (h1 + h2 + h3 + h4 + h5), ha="center", fontsize=15)

plt.xlabel("Different Activations")

plt.ylabel("Time splits")

plt.title(f"Time Splits of Different Activations, {test_case}, {slice_a}-{slice_b}, {run_tag}")

plt.legend()

saving_dir = "steady_turbine/diffacts_timesplits"
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

plt.savefig(f"{saving_dir}/diffacts_timesplits_{test_case}_{slice_a}-{slice_b}{run_tag}_withTotTime")

