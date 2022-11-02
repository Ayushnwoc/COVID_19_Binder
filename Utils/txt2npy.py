import numpy as np
import sys

"""
Python script to convert contact map from txt to npy
"""
name = sys.argv[1] # name of the protein
path = sys.argv[2] # path to contact map txt file

with open(path, 'r') as file:
    data = file.read()

save_path = name + '.npy'

out_list = data.split('\n')

for i in range(len(out_list)):

    out_list[i] = out_list[i].split(' ')

out_list = out_list[0:-1]
out_arr = np.array(out_list, dtype=float)

np.save(save_path, out_arr)