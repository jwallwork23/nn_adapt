# from nn_adapt.features import *
# from nn_adapt.features import extract_array
# from nn_adapt.metric import *
# from nn_adapt.parse import Parser
# from nn_adapt.solving_one2n import *
# from nn_adapt.solving_n2n import *
# from nn_adapt.solving import *
# from nn_adapt.utility import ConvergenceTracker
# from firedrake.meshadapt import adapt
# from firedrake.petsc import PETSc

# import importlib
# import numpy as np

# tt_steps = 10

# # setup1 = importlib.import_module(f"burgers_n2n.config")
# # meshes = [UnitSquareMesh(20, 20) for _ in range(tt_steps)]
# # out1 = indicate_errors_n2n(meshes=meshes, config=setup1)
# # print(out1)

# mesh = UnitSquareMesh(20, 20)
# setup2 = importlib.import_module(f"burgers_one2n.config")
# out2 = indicate_errors_one2n(mesh=mesh, config=setup2)
# print(out2)

# # mesh = UnitSquareMesh(20, 20)
# # setup2 = importlib.import_module(f"burgers_one2n.config")
# # out2 = get_solutions_one2n(mesh=mesh, config=setup2)
# # fwd_sol = out2["forward"]


# lines = [[1,6,8,5], [1,3,7,6,5], [2,8,5]]
# length = 3

# end = 5
# id_list = [0 for _ in range(length)]
# toend = 0
# steps = 0
# while not toend:
#     steps += 1
#     forward = [1 for _ in range(length)]
#     t = [lines[id][item] for id, item in enumerate(id_list)]
#     toend = 1
#     for item in t:
#         toend = 0 if item != end else 1
#     if toend == 1:
#         break;
    
#     for id, item in enumerate(t):
#         for line_id, line in enumerate(lines):
#             if item in line[id_list[line_id]+1:]:
#                 forward[id] = 0
#                 break;
#     for i in range(length):
#         id_list[i] += forward[i]
    
    
# print(steps)    


# def dec2bin(input):
#     return "{0:b}".format(input)

# def bin2dec(input):
#     length = len(input)
#     output = 0
#     for id, item in enumerate(input):
#         output += pow(2, length-1-id) * int(item)
#     return output

# def sp_add(input1, input2):
#     min_len = min(len(input1), len(input2))
#     max_len = max(len(input1), len(input2))
#     input1 = input1[::-1]
#     input2 = input2[::-1]
    
#     output = ""
#     for i in range(max_len):
#         if i < min_len:
#             if input1[i] == input2[i]:
#                 output += "0"
#             else:
#                 output += "1"
#         else:
#             try:
#                 output += input1[i]
#             except:
#                 output += input2[i]
    
#     return output[::-1]

# a = dec2bin(9)
# b = dec2bin(5)
# c = sp_add(a, b)
# print(a, b, c)

a = [1,1,1,1]
for i, j in enumerate(a):
    print(j)

