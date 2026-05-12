#!/usr/bin/env python
# -*- coding:utf-8 -*-

from operator import rshift
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sys import argv #swit = int(argv[1])
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel

import analysis_data_distribution as add
import galaxy_models as gm
import fit_rho_fJ as fff
import integrate_methods as im



if __name__ == '__main__':

    modelname = ["_original", "_M1e-1", "_M1e-2", "_N1e1", ""]
    modelId = 0
    is_centerize = 1
    rs = 19.6

    levels = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 1.]
    N_levels = len(levels)
    # descriptions = ["0.001", "0.01", "0.05", "0.1", "0.3", "0.5", "0.9", "1."]
    descriptions = ["%.2f%%"%(levels[j]*100) for j in np.arange(N_levels)]

    N_snapshots = 90
    d_snapshots = 10
    snapshots = [i for i in np.arange(N_snapshots)*d_snapshots]
    t = np.arange(N_snapshots)*0.01*d_snapshots
    LagRadii = np.zeros((N_snapshots, N_levels))

    for i in np.arange(N_snapshots):
        ss = snapshots[i]
        filename = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general%s/snaps/txt/snapshot_%03d.txt" % (modelname[modelId], ss)
        data = np.array(np.loadtxt(filename, dtype=float))
        x = data[:, 0:3]
        v = data[:, 3:6]
        if is_centerize:
            for k in range(3):
                x[:,k] -= np.median(x[:,k])
                v[:,k] -= np.mean(v[:,k])
        r = add.norm_l(x, axis=1)
        r_sort = np.sort(r)
        N_ptcs = len(r)
        for j in np.arange(N_levels):
            LagRadii[i,j] = r_sort[int(N_ptcs*levels[j])-1]/rs
        print(r"Reading %s ... Done."%(filename))

    fig = plt.figure()
    pointsize = 2.
    fontsize = 18.0
    plt.subplot(1,1,1)
    for j in np.arange(N_levels):
        plt.plot(t, LagRadii[:,j], lw=pointsize, label=r"%s"%(descriptions[j]))
        # plt.scatter(x, y, s=pointsize)
    plt.xlabel(r"eloluiton time (Gyr)", fontsize=fontsize)
    plt.ylabel(r"Lagrangian radii (logrithm rate to scale length)", fontsize=fontsize)
    plt.legend(fontsize=fontsize/2)
    plt.yscale("log")
    # whspace = 0.4
    # plt.subplots_adjust(hspace=whspace, wspace=whspace)
    plt.suptitle(r"galaxy_count1e4_NFW_spherical_rs19.6_mass%s"%(modelname[modelId]))
    plt.show()
    plt.close()
    