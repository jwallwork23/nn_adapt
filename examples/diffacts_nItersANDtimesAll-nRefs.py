import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description="Specify test case, run tag, nRef slicing and specific task.")
parser.add_argument("--test_case", 
                    help="aligned, offset, aligned_reversed or trench",
                    type=str,
                    default="aligned")
parser.add_argument("--run_tag", 
                    help="nothing, _run1, _run_2, 300, 400, etc.",
                    type=str,
                    default="")
parser.add_argument("--nrefs_start", 
                    help="start (pythonic) slice to [1, 2, ... , 24, 25], pay attention to reversed case",
                    type=int,
                    default="0")
parser.add_argument("--nrefs_end", 
                    help="end (pythonic) slice to [1, 2, ... , 24, 25], pay attention to reversed case",
                    type=int,
                    default="25")
parser.add_argument("--spec_task", 
                    help="diffActs_nIters-nRefs, diffActs_timesAll-nRefs",
                    type=str,
                    default="nIters-nRefs")


args = parser.parse_args()
test_case = args.test_case
run_tag = args.run_tag
nrefs_start = args.nrefs_start
nrefs_end = args.nrefs_end
spec_task = args.spec_task



nRefs = len(np.load("steady_turbine/data/niter_GOanisotropic_" + test_case + run_tag + ".npy"))
refs = np.arange(1, nRefs + 1)[nrefs_start : nrefs_end] # SLICED



if spec_task == "diffActs_nIters-nRefs":
    
    saving_dir = "steady_turbine/diffacts_nIters-nRefs"
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    # for aligned reversed case, len(go) = 24 but len(sigm) = 25
    if test_case == "aligned_reversed":
        # get original length
        nRefs_GO = len(np.load("steady_turbine/data/niter_GOanisotropic_" + test_case + run_tag + ".npy"))
        nRefs_ML = len(np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy"))

        niter_GO = np.load("steady_turbine/data/niter_GOanisotropic_" + test_case + run_tag + ".npy")[nrefs_start : nrefs_end] # SLICED 
        niter_sigm = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy")[nrefs_start : nrefs_end] # SLICED
        niter_tanh = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_2_F_64_100-4" + run_tag + ".npy")[nrefs_start : nrefs_end] # SLICED
        niter_relu = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_3_F_64_100-4" + run_tag + ".npy")[nrefs_start : nrefs_end]# SLICED

        refs_GO = np.arange(1, nRefs_GO + 1)[nrefs_start : nrefs_end] # SLICED
        refs_ML = np.arange(1, nRefs_ML + 1)[nrefs_start : nrefs_end] # SLICED

        # plot
        plt.plot(refs_GO, niter_GO, label="GO") # NOTE X AND Y HERE ALREADY SLICED
        plt.plot(refs_ML, niter_sigm, label="sigm")
        plt.plot(refs_ML, niter_tanh, label="tanh")
        plt.plot(refs_ML, niter_relu, label="relu")
        plt.xlabel("Refinements")
        plt.ylabel("Number of Iterations in Each Refinement")
        plt.title(f"Number of Iterations vs Refinements of different activations, {test_case}, {run_tag}, {nrefs_start + 1}-{nrefs_end}")
        plt.legend()
        plt.savefig(f"{saving_dir}/nIters-nRefs_{test_case}{run_tag}_{nrefs_start + 1}-{nrefs_end}")
        plt.close()

    else:
        niter_GO = np.load("steady_turbine/data/niter_GOanisotropic_" + test_case + run_tag + ".npy")[nrefs_start : nrefs_end] # SLICED 
        niter_sigm = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy")[nrefs_start : nrefs_end] # SLICED
        niter_tanh = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_2_F_64_100-4" + run_tag + ".npy")[nrefs_start : nrefs_end] # SLICED
        niter_relu = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_3_F_64_100-4" + run_tag + ".npy")[nrefs_start : nrefs_end]# SLICED

        # plot
        plt.plot(refs, niter_GO, label="GO") # NOTE X AND Y HERE ALREADY SLICED
        plt.plot(refs, niter_sigm, label="sigm")
        plt.plot(refs, niter_tanh, label="tanh")
        plt.plot(refs, niter_relu, label="relu")
        plt.xlabel("Number of refinements")
        plt.ylabel("Number of Iterations in Each Refinement")
        plt.title(f"Number of Iterations vs Refinements of different activations, {test_case}, {run_tag}, {nrefs_start + 1}-{nrefs_end}")
        plt.legend()
        plt.savefig(f"{saving_dir}/nIters-nRefs_{test_case}{run_tag}_{nrefs_start + 1}-{nrefs_end}")
        plt.close



if spec_task == "diffActs_timesAll-nRefs":
    
    
    saving_dir = "steady_turbine/diffacts_timesAll-nRefs"
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    if test_case == "aligned_reversed":
        
        nRefs_GO = len(np.load("steady_turbine/data/times_all_GOanisotropic_" + test_case + run_tag + ".npy"))
        nRefs_ML = len(np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy"))

        # timesAll
        times_GO = np.load("steady_turbine/data/times_all_GOanisotropic_" + test_case + run_tag + ".npy")[nrefs_start : nrefs_end]
        times_sigm = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy")[nrefs_start : nrefs_end]
        times_tanh = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4_Tanh" + run_tag + ".npy")[nrefs_start : nrefs_end]
        times_relu = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4_ReLU" + run_tag + ".npy")[nrefs_start : nrefs_end]

        refs_GO = np.arange(1, nRefs_GO + 1)[nrefs_start : nrefs_end] # SLICED
        refs_ML = np.arange(1, nRefs_ML + 1)[nrefs_start : nrefs_end] # SLICED

        # NOTE: times_all and sum(times_part) are a bit different

        # plot
        plt.plot(refs_GO, times_GO, label="GO")
        plt.plot(refs_ML, times_sigm, label="sigm")
        plt.plot(refs_ML, times_tanh, label="tanh")
        plt.plot(refs_ML, times_relu, label="relu")
        plt.xlabel("Refinements")
        plt.ylabel("Whole Times")
        plt.title(f"Whole Times vs Refinements of different activations, {test_case}{run_tag}_{nrefs_start}-{nrefs_end}")
        plt.legend()
        plt.savefig(f"{saving_dir}/wholeTimes-nRefs_{test_case}_{nrefs_start}-{nrefs_end}{run_tag}")
        plt.close()

    else:

        # timesAll
        times_GO = np.load("steady_turbine/data/times_all_GOanisotropic_" + test_case + run_tag + ".npy")[nrefs_start : nrefs_end]
        times_sigm = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy")[nrefs_start : nrefs_end]
        times_tanh = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4_Tanh" + run_tag + ".npy")[nrefs_start : nrefs_end]
        times_relu = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4_ReLU" + run_tag + ".npy")[nrefs_start : nrefs_end]

        # NOTE: times_all and sum(times_part) are a bit different

        # plot
        plt.plot(refs, times_GO, label="GO")
        plt.plot(refs, times_sigm, label="sigm")
        plt.plot(refs, times_tanh, label="tanh")
        plt.plot(refs, times_relu, label="relu")
        plt.xlabel("Refinements")
        plt.ylabel("Whole Times")
        plt.title(f"Whole Times vs Refinements of different activations, {test_case}{run_tag}_{nrefs_start}-{nrefs_end}")
        plt.legend()
        plt.savefig(f"{saving_dir}/wholeTimes-nRefs_{test_case}_{nrefs_start}-{nrefs_end}{run_tag}")
        plt.close()