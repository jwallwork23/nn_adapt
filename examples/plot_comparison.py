from cgi import test
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

#====================================== Terminal Argument Parsing ==================================
# NHLs: Number of Hidden Layers
# refs: Refinements
# nRefs: total number of refinements
parser = argparse.ArgumentParser(description="specify task, test case, run tag and refinement slicing.")
parser.add_argument("--spec_task", 
                    help="NHLs_qoiErrors-refs, avgQoIErrors-NHLs, NHLs_nIters-refs, NHLs_timesAll-refs, timeSplits-NHLs, acts_qoiErrors-refs, avgQoIErrors-acts, acts_nIters-refs, acts_timesAll-refs, timeSplits-acts",
                    type=str,
                    default="NHLs_qoiError-refs")
parser.add_argument("--test_case", 
                    help="aligned, offset, aligned_reversed, trench",
                    type=str,
                    default="aligned")
parser.add_argument("--run_tag", 
                    help="nothing, _run1, _run_2, 300, 400, etc.",
                    type=str,
                    default="")
parser.add_argument("--nRefs_start", 
                    help="start (pythonic) slice to [1, 2, ... , 22, 23, 24, 25]",
                    type=int,
                    default="19")
parser.add_argument("--nRefs_end", 
                    help="end (pythonic) slice to [1, 2, ... , 22, 23, 24, 25]",
                    type=int,
                    default="25")

args = parser.parse_args()
spec_task = args.spec_task
test_case = args.test_case
run_tag = args.run_tag
nRefs_start = args.nRefs_start
nRefs_end = args.nRefs_end


#======================================== Global Variables =====================================
NHLs = ["ML1", "ML2", "ML3", "ML4", "ML5", "ML6", "ML7", "ML8", "ML9", "ML10"]
NHLss = ["GO"] + NHLs

parts = ["forward", "adjoint", "estimator", "metric", "adapt"]
partss = ["all"] + parts

acts = ["Sigmoid", "Tanh", "ReLU"]
actss = ["GO"] + acts

# get nRefs automatically from goal-oriented QoIs
nRefs = len(np.load(f"steady_turbine/data/qois_GOanisotropic_{test_case}{run_tag}.npy"))
refs = np.arange(1, nRefs + 1)[nRefs_start : nRefs_end] # SLICED HERE

saving_dir = "steady_turbine/plot_comparison"
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)


#======================================== NHLs_qoiErrors-refs & avgQoIError-NHLs =============================================
if spec_task in ["NHLs_qoiErrors-refs", "avgQoIErrors-NHLs"]:
    
    # load raw data and slice
    qois = {}
    qois["GO"] = np.load(f"steady_turbine/data/qois_GOanisotropic_{test_case}{run_tag}.npy")[nRefs_start : nRefs_end]
    for NHLi in NHLs:
        qois[NHLi] = np.load(f"steady_turbine/data/qois_MLanisotropic_{test_case}_" + NHLi[2:] + f"_F_64_100-4{run_tag}.npy")[nRefs_start : nRefs_end]

    conv = np.load(f"steady_turbine/data/qois_uniform_{test_case}.npy")[-1]

    errors = {}
    for NHLii in NHLss:
        errors[NHLii] = np.abs((qois[NHLii] - conv) / conv)
      
#---------------------------------------- NHLs_qoiErrors-refs ---------------------------------------------
    if spec_task == "NHLs_qoiErrors-refs":

        # plot
        for NHLii in NHLss:
            plt.plot(refs, errors[NHLii], label=NHLii) # DATA ALREADY SLICED
        xl = "Refinements"
        yl = "QoI Errors"
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.title(f"{yl} vs {xl}, {test_case}{run_tag}")
        plt.legend()
        plt.savefig(f"{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}")
        print(f"Figure [{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}] saved successfully.")
        plt.close()

#---------------------------------------- avgQoIErrors-NHLs ---------------------------------------------
    if spec_task == "avgQoIErrors-NHLs":
    
        errors_avg = {}
        for NHLii in NHLss:
            errors_avg[NHLii] = np.mean(errors[NHLii])
        # plot 
        stk1 = plt.bar(errors_avg.keys(), errors_avg.values())
        for s1 in stk1:
            h1 = s1.get_height()
            plt.text(s1.get_x() + s1.get_width() / 2., h1 / 2., "%.5f" % h1, ha="center", fontsize=6)
        xl = "Number of Hidden Layers"
        yl = "Average QoI Errors"
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.title(f"{yl} vs {xl}, {test_case}{run_tag}")
        plt.savefig(f"{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}")
        print(f"Figure [{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}] saved successfully.")
        plt.close()


#======================================== NHLs_nIters-refs =============================================
if spec_task == "NHLs_nIters-refs":

    # load raw data and slice
    nIters = {}
    nIters["GO"] = np.load(f"steady_turbine/data/niter_GOanisotropic_{test_case}{run_tag}.npy")[nRefs_start : nRefs_end]
    for NHLi in NHLs:
        nIters[NHLi] = np.load(f"steady_turbine/data/niter_MLanisotropic_{test_case}_{NHLi[2:]}_F_64_100-4{run_tag}.npy")[nRefs_start : nRefs_end]
    
    # plot
    for NHLii in NHLss:
        plt.plot(refs, nIters[NHLii], label=NHLii)
    xl = "Refinements"
    yl = "Total Number of Iterations"
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(f"{yl} vs {xl}, {test_case}{run_tag}")
    plt.legend()
    plt.savefig(f"{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}")
    print(f"Figure [{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}] saved successfully.")
    plt.close()



#======================================== NHLs_timesAll-refs & timeSplits-NHLs =============================================
if spec_task in ["NHLs_timesAll-refs", "timeSplits-NHLs"] :
    
    # load raw data
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

    times_GO["all"] = np.load("steady_turbine/data/times_all_GOanisotropic_" + test_case + run_tag + ".npy")[nRefs_start : nRefs_end]
    times_ML1["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
    times_ML2["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_2_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
    times_ML3["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_3_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
    times_ML4["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_4_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
    times_ML5["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_5_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
    times_ML6["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_6_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
    times_ML7["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_7_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
    times_ML8["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_8_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
    times_ML9["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_9_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
    times_ML10["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_10_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]

    for part in parts:
        times_GO[part] = np.load("steady_turbine/data/times_" + part + "_GOanisotropic_" + test_case + run_tag + ".npy")[nRefs_start : nRefs_end]
        times_ML1[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
        times_ML2[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_2_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
        times_ML3[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_3_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
        times_ML4[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_4_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
        times_ML5[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_5_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
        times_ML6[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_6_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
        times_ML7[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_7_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
        times_ML8[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_8_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
        times_ML9[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_9_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
        times_ML10[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_10_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]

    # aggregate new data
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

    for partii in partss:
        times_GO_avg[partii] = np.mean(times_GO[partii])
        times_ML1_avg[partii] = np.mean(times_ML1[partii])
        times_ML2_avg[partii] = np.mean(times_ML2[partii])
        times_ML3_avg[partii] = np.mean(times_ML3[partii])
        times_ML4_avg[partii] = np.mean(times_ML4[partii])
        times_ML5_avg[partii] = np.mean(times_ML5[partii])
        times_ML6_avg[partii] = np.mean(times_ML6[partii])
        times_ML7_avg[partii] = np.mean(times_ML7[partii])
        times_ML8_avg[partii] = np.mean(times_ML8[partii])
        times_ML9_avg[partii] = np.mean(times_ML9[partii])
        times_ML10_avg[partii] = np.mean(times_ML10[partii])

#---------------------------------------- NHLs_timesAll-refs ---------------------------------------------
    if spec_task == "NHLs_timesAll-refs": 

        # plot
        plt.plot(refs, times_GO["all"], label="GO")
        plt.plot(refs, times_ML1["all"], label="ML1")
        plt.plot(refs, times_ML2["all"], label="ML2")
        plt.plot(refs, times_ML3["all"], label="ML3")
        plt.plot(refs, times_ML4["all"], label="ML4")
        plt.plot(refs, times_ML5["all"], label="ML5")
        plt.plot(refs, times_ML6["all"], label="ML6")
        plt.plot(refs, times_ML7["all"], label="ML7")
        plt.plot(refs, times_ML8["all"], label="ML8")
        plt.plot(refs, times_ML9["all"], label="ML9")
        plt.plot(refs, times_ML10["all"], label="ML10")
        xl = "Refinements"
        yl = "Whole Times"
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.title(f"{yl} vs {xl}, {test_case}{run_tag}")
        plt.legend()
        plt.savefig(f"{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}")
        print(f"Figure [{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}] saved successfully.")
        plt.close()

#---------------------------------------- timeSplits-NHLs ---------------------------------------------
    if spec_task == "timeSplits-NHLs":
        
        # agg agg new data
        mat = np.stack((
                list(times_GO_avg.values()),
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

        # plot
        stk1 = plt.bar(NHLss, forward, 0.5, label="forward")
        stk2 = plt.bar(NHLss, adjoint, 0.5, label="adjoint", bottom=forward)
        stk3 = plt.bar(NHLss, estimator, 0.5, label="estimator", bottom=forward+adjoint)
        stk4 = plt.bar(NHLss, metric, 0.5, label="metric", bottom=forward+adjoint+estimator)
        stk5 = plt.bar(NHLss, adapt, 0.5, label="adapt", bottom=forward+adjoint+estimator+metric)

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

        xl = "Number of Hidden Layers"
        yl = "Time Splits"
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.title(f"{yl} vs {xl}, {test_case}{run_tag}")
        plt.legend()
        plt.savefig(f"{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}")
        print(f"Figure [{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}] saved successfully.")
        plt.close()


#======================================== acts_qoiErrors-refs & avgQoIErrors-acts =============================================
if spec_task in ["acts_qoiErrors-refs", "avgQoIErrors-acts"]:

    # load raw data and slice
    qois = {}
    qois["GO"] = np.load(f"steady_turbine/data/qois_GOanisotropic_{test_case}{run_tag}.npy")[nRefs_start : nRefs_end]
    qois["Sigmoid"] = np.load(f"steady_turbine/data/qois_MLanisotropic_{test_case}_1_F_64_100-4{run_tag}.npy")[nRefs_start : nRefs_end]
    qois["Tanh"] = np.load(f"steady_turbine/data/qois_MLanisotropic_{test_case}_1_F_64_100-4_Tanh{run_tag}.npy")[nRefs_start : nRefs_end]
    qois["ReLU"] = np.load(f"steady_turbine/data/qois_MLanisotropic_{test_case}_1_F_64_100-4_ReLU{run_tag}.npy")[nRefs_start : nRefs_end]

    conv = np.load(f"steady_turbine/data/qois_uniform_{test_case}.npy")[-1]

    # agg new data
    errors = {}
    for actii in actss:
        errors[actii] = np.abs((qois[actii] - conv) / conv)
    
#---------------------------------------- acts_qoiErrors-refs ---------------------------------------------
    if spec_task == "acts_qoiErrors-refs":

        # plot
        for actii in actss:
            plt.plot(refs, errors[actii], label=actii) # DATA ALREADY SLICED
        xl = "Refinements"
        yl = "QoI Errors"
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.title(f"{yl} vs {xl}, {test_case}{run_tag}")
        plt.legend()
        plt.savefig(f"{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}")
        print(f"Figure [{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}] saved successfully.")
        plt.close()
        
#---------------------------------------- avgQoIErrors-acts ---------------------------------------------
    if spec_task == "avgQoIErrors-acts":
        
        # agg new new data
        errors_avg = {}
        for actii in actss:
            errors_avg[actii] = np.mean(errors[actii])

        # plot
        stk1 = plt.bar(errors_avg.keys(), errors_avg.values())
        for s1 in stk1:
            h1 = s1.get_height()
            plt.text(s1.get_x() + s1.get_width() / 2., h1 / 2., "%.5f" % h1, ha="center")
        xl = "Activations"
        yl = "Average QoI Errors"
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.title(f"{yl} vs {xl}, {test_case}{run_tag}")
        plt.savefig(f"{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}")
        print(f"Figure [{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}] saved successfully.")
        plt.close()

        '''stk5 = plt.bar(actss, adapt, 0.5, label="adapt", bottom=forward+adjoint+estimator+metric)

    for s1, s2, s3, s4, s5 in zip(stk1, stk2, stk3, stk4, stk5):
        h1 = s1.get_height()
        plt.text(s1.get_x() + s1.get_width() / 2., h1 / 2., "%.2f" % h1, ha="center")'''


#======================================== acts_nIters-refs =============================================
if spec_task == "acts_nIters-refs":

    if test_case == "aligned_reversed":

        # get original length
        nRefs_GO = len(np.load("steady_turbine/data/niter_GOanisotropic_" + test_case + run_tag + ".npy"))
        nRefs_ML = len(np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy"))
        refs_GO = np.arange(1, nRefs_GO + 1)[nRefs_start : nRefs_end] # SLICED
        refs_ML = np.arange(1, nRefs_ML + 1)[nRefs_start : nRefs_end] # SLICED

        # load raw data
        niter_GO = np.load("steady_turbine/data/niter_GOanisotropic_" + test_case + run_tag + ".npy")[nRefs_start : nRefs_end] # SLICED 
        niter_sigm = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end] # SLICED
        niter_tanh = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_1_F_64_100-4_Tanh" + run_tag + ".npy")[nRefs_start : nRefs_end] # SLICED
        niter_relu = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_1_F_64_100-4_ReLU" + run_tag + ".npy")[nRefs_start : nRefs_end]# SLICED

        # plot
        plt.plot(refs_GO, niter_GO, label="GO") # NOTE X AND Y HERE ALREADY SLICED
        plt.plot(refs_ML, niter_sigm, label="sigm")
        plt.plot(refs_ML, niter_tanh, label="tanh")
        plt.plot(refs_ML, niter_relu, label="relu")
        xl = "Refinements"
        yl = "Number of Iterations"
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.title(f"{yl} vs {xl}, {test_case}{run_tag}")
        plt.legend()
        plt.savefig(f"{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}")
        print(f"Figure [{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}] saved successfully.")
        plt.close()

    else:
        niter_GO = np.load("steady_turbine/data/niter_GOanisotropic_" + test_case + run_tag + ".npy")[nRefs_start : nRefs_end] # SLICED 
        niter_sigm = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end] # SLICED
        niter_tanh = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_1_F_64_100-4_Tanh" + run_tag + ".npy")[nRefs_start : nRefs_end] # SLICED
        niter_relu = np.load("steady_turbine/data/niter_MLanisotropic_" + test_case + "_1_F_64_100-4_ReLU" + run_tag + ".npy")[nRefs_start : nRefs_end]# SLICED

        # plot
        plt.plot(refs, niter_GO, label="GO") # NOTE X AND Y HERE ALREADY SLICED
        plt.plot(refs, niter_sigm, label="sigm")
        plt.plot(refs, niter_tanh, label="tanh")
        plt.plot(refs, niter_relu, label="relu")
        xl = "Refinements"
        yl = "Number of Iterations"
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.title(f"{yl} vs {xl}, {test_case}{run_tag}")
        plt.legend()
        plt.savefig(f"{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}")
        print(f"Figure [{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}] saved successfully.")
        plt.close()


#======================================== acts_timesAll-refs =============================================
if spec_task == "acts_timesAll-refs":
    
    if test_case == "aligned_reversed":
        
        nRefs_GO = len(np.load("steady_turbine/data/times_all_GOanisotropic_" + test_case + run_tag + ".npy"))
        nRefs_ML = len(np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy"))

        refs_GO = np.arange(1, nRefs_GO + 1)[nRefs_start : nRefs_end] # SLICED
        refs_ML = np.arange(1, nRefs_ML + 1)[nRefs_start : nRefs_end] # SLICED

        # load raw data
        times_GO = np.load("steady_turbine/data/times_all_GOanisotropic_" + test_case + run_tag + ".npy")[nRefs_start : nRefs_end]
        times_sigm = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
        times_tanh = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4_Tanh" + run_tag + ".npy")[nRefs_start : nRefs_end]
        times_relu = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4_ReLU" + run_tag + ".npy")[nRefs_start : nRefs_end]

        # NOTE: times_all and sum(times_part) are a bit different

        # plot
        plt.plot(refs_GO, times_GO, label="GO")
        plt.plot(refs_ML, times_sigm, label="sigm")
        plt.plot(refs_ML, times_tanh, label="tanh")
        plt.plot(refs_ML, times_relu, label="relu")
        xl = "Refinements"
        yl = "Whole Times"
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.title(f"{yl} vs {xl}, {test_case}{run_tag}")
        plt.legend()
        plt.savefig(f"{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}")
        print(f"Figure [{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}] saved successfully.")
        plt.close()

    else:

        # load raw data
        times_GO = np.load("steady_turbine/data/times_all_GOanisotropic_" + test_case + run_tag + ".npy")[nRefs_start : nRefs_end]
        times_sigm = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
        times_tanh = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4_Tanh" + run_tag + ".npy")[nRefs_start : nRefs_end]
        times_relu = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4_ReLU" + run_tag + ".npy")[nRefs_start : nRefs_end]

        # NOTE: times_all and sum(times_part) are a bit different

        # plot
        plt.plot(refs, times_GO, label="GO")
        plt.plot(refs, times_sigm, label="sigm")
        plt.plot(refs, times_tanh, label="tanh")
        plt.plot(refs, times_relu, label="relu")
        xl = "Refinements"
        yl = "Whole Times"
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.title(f"{yl} vs {xl}, {test_case}{run_tag}")
        plt.legend()
        plt.savefig(f"{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}")
        print(f"Figure [{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}] saved successfully.")
        plt.close()


#======================================== timeSplits-acts =============================================
if spec_task == "timeSplits-acts":

    # load raw data and slice
    times_GO = {}
    times_sigm = {}
    times_relu = {}
    times_tanh = {}

    times_GO["all"] = np.load("steady_turbine/data/times_all_GOanisotropic_" + test_case + run_tag + ".npy")[nRefs_start : nRefs_end]
    times_sigm["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
    times_tanh["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4_Tanh" + run_tag + ".npy")[nRefs_start : nRefs_end]
    times_relu["all"] = np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_1_F_64_100-4_ReLU" + run_tag + ".npy")[nRefs_start : nRefs_end]
    
    for part in parts:
        times_GO[part] = np.load("steady_turbine/data/times_" + part + "_GOanisotropic_" + test_case + run_tag + ".npy")[nRefs_start : nRefs_end]
        times_sigm[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_1_F_64_100-4" + run_tag + ".npy")[nRefs_start : nRefs_end]
        times_tanh[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_1_F_64_100-4_Tanh" + run_tag + ".npy")[nRefs_start : nRefs_end]
        times_relu[part] = np.load("steady_turbine/data/times_" + part + "_MLanisotropic_" + test_case + "_1_F_64_100-4_ReLU" + run_tag + ".npy")[nRefs_start : nRefs_end]
        
    # agg new data
    times_GO_avg = {}
    times_sigm_avg = {}
    times_tanh_avg = {}
    times_relu_avg = {}

    for partii in partss:
        times_GO_avg[partii] = np.mean(times_GO[partii])
        times_sigm_avg[partii] = np.mean(times_sigm[partii])
        times_tanh_avg[partii] = np.mean(times_tanh[partii])
        times_relu_avg[partii] = np.mean(times_relu[partii])

    # stacked bars of time splits (avraged over 20~25th refinements)
    mat = np.stack((list(times_GO_avg.values()),
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


    stk1 = plt.bar(actss, forward, 0.5, label="forward")
    stk2 = plt.bar(actss, adjoint, 0.5, label="adjoint", bottom=forward)
    stk3 = plt.bar(actss, estimator, 0.5, label="estimator", bottom=forward+adjoint)
    stk4 = plt.bar(actss, metric, 0.5, label="metric", bottom=forward+adjoint+estimator)
    stk5 = plt.bar(actss, adapt, 0.5, label="adapt", bottom=forward+adjoint+estimator+metric)

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

    xl = "Activations"
    yl = "Time Splits"
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(f"{yl} vs {xl}, {test_case}{run_tag}")
    plt.legend()
    plt.savefig(f"{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}")
    print(f"Figure [{saving_dir}/{spec_task}_{test_case}{run_tag}_{nRefs_start + 1}-{nRefs_end}] saved successfully.")
    plt.close()