#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import analysis_data_distribution as add
import galaxy_models as gm
import fit_rho_fJ as fff

# main()
if __name__ == '__main__':

    filename = "/home/darkgaia/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/snapshot_200.secondhand_005.txt"
    dataread = np.loadtxt(filename, dtype=float)
    JJJ = dataread[:, 6:9]
    J_normalrange = abs(JJJ) #screen
    fJ = dataread[:, -3]+1.e-200
    comb_sum11 = add.norm_l(J_normalrange, axis=1)
    # comb_sum11 = np.sum(J_normalrange, axis=1)/3
    # comb_sum11 = (abs(JJJ[:,0])+abs(JJJ[:,1])+abs(JJJ[:,2]))/3

    plt.scatter(comb_sum11, fJ)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
