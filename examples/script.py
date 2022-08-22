import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Specify run tag.")
parser.add_argument("--test_case", 
                    help="aligned, offset, aligned_reversed, trench",
                    type=str,
                    default="aligned")
parser.add_argument("--tag", 
                    help="1_F_64_100-4, etc.",
                    type=str,
                    default="1_F_64_100-4")        
parser.add_argument("--run_num", 
                    help="Nothing, _run1, _run_2, 300, 400, 500, etc.",
                    type=str,
                    default="")                 
args = parser.parse_args()
test_case = args.test_case
tag = args.tag
run_num = args.run_num

print(np.load("steady_turbine/data/times_all_MLanisotropic_" + test_case + "_" + tag + run_num + ".npy"))