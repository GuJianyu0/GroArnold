#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import analysis_data_distribution as add

def print_info_gm(filename, tag=1):
    data = np.loadtxt(filename)
    x = data[:,0:3]
    v = data[:,3:6]
    X = add.norm_l(x, axis=1)
    V = add.norm_l(v, axis=1)
    add.DEBUG_PRINT_V(1, np.shape(data))
    add.DEBUG_PRINT_V(1, [x[0,0], np.mean(x[:,0]), np.median(X)], "X")
    add.DEBUG_PRINT_V(1, [v[0,0], np.mean(v[:,0]), np.median(V)], "V")
    return 0

if __name__ == "__main__":

    filename1 = "/home/darkgaia/workroom/0prog/sandbox/"\
        "GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/"\
        "txt/snapshot_000.txt"
    filename11 = "/home/darkgaia/workroom/0prog/sandbox/"\
        "GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/"\
        "txt/snapshot_000_preprocessed.txt"
    filename2 = "/home/darkgaia/workroom/0prog/sandbox/"\
        "GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/"\
        "aa/snapshot_0.action.method_all.txt"
    
    # filename1 = "/home/darkgaia/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general_20220717/"\
    #     "txt/snapshot_003.txt"
    # filename11 = "/home/darkgaia/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general_20220717/"\
    #     "txt/snapshot_003_preprocessed.txt"
    # filename2 = "/home/darkgaia/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general_20220717/"\
    #     "aa/snapshot_3.action.method_all.txt"
    
    print_info_gm(filename1)
    print_info_gm(filename11)
    # print_info_gm(filename2)
