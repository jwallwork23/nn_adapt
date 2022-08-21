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

times_GO["all"] = np.load("steady_turbine/data/times_all_GOanisotropic_" + test_case + run_tag + ".npy")[19:]
times_ML1["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy")[19:]
times_ML2["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_2_F_64_100-4" + run_tag + ".npy")[19:]
times_ML3["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_3_F_64_100-4" + run_tag + ".npy")[19:]
times_ML4["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_4_F_64_100-4" + run_tag + ".npy")[19:]
times_ML5["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_5_F_64_100-4" + run_tag + ".npy")[19:]
times_ML6["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_6_F_64_100-4" + run_tag + ".npy")[19:]
times_ML7["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_7_F_64_100-4" + run_tag + ".npy")[19:]
times_ML8["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_8_F_64_100-4" + run_tag + ".npy")[19:]
times_ML9["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_9_F_64_100-4" + run_tag + ".npy")[19:]
times_ML10["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_10_F_64_100-4" + run_tag + ".npy")[19:]

for part in parts:
    times_GO[part] = np.load("steady_turbine/data/times_" + part + "_GOanisotropic_" + test_case + run_tag + ".npy")[19:]
    times_ML1[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy")[19:]
    times_ML2[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_2_F_64_100-4" + run_tag + ".npy")[19:]
    times_ML3[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_3_F_64_100-4" + run_tag + ".npy")[19:]
    times_ML4[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_4_F_64_100-4" + run_tag + ".npy")[19:]
    times_ML5[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_5_F_64_100-4" + run_tag + ".npy")[19:]
    times_ML6[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_6_F_64_100-4" + run_tag + ".npy")[19:]
    times_ML7[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_7_F_64_100-4" + run_tag + ".npy")[19:]
    times_ML8[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_8_F_64_100-4" + run_tag + ".npy")[19:]
    times_ML9[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_9_F_64_100-4" + run_tag + ".npy")[19:]
    times_ML10[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_10_F_64_100-4" + run_tag + ".npy")[19:]

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

# stacked bars of time splits (avraged over 20~25th refinements)
methods = ["GO", "ML1", "ML2", "ML3", "ML4", "ML5", "ML6", "ML7", "ML8", "ML9", "ML10"]

mat = np.stack((list(times_GO_avg.values()),
                          list(times_ML1_avg.values()),
                          list(times_ML2_avg.values()),
                          list(times_ML3_avg.values()),
                          list(times_ML4_avg.values()),
                          list(times_ML5_avg.values()),
                          list(times_ML6_avg.values()),
                          list(times_ML7_avg.values()),
                          list(times_ML8_avg.values()),
                          list(times_ML9_avg.values()),
                          list(times_ML10_avg.values()),
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

plt.title("Time Splits of Different Networks of " + test_case + run_tag)

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

plt.xlabel("Different Networks")

plt.ylabel("Time splits")

plt.title(f"Time Splits of Different Networks, {test_case}, {run_tag}")

plt.legend()

saving_dir = "steady_turbine/diffnets_plots_timesplits"
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

plt.savefig(f"{saving_dir}/time_splits_of_diffnets_{test_case}{run_tag}")