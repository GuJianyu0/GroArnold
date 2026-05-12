#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import analysis_data_distribution as add
import galaxy_models as gm
import fit_rho_fJ as fff



if __name__ == '__main__':

    filename_direct_tp_orb = "/home/darkgaia/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/action_direct_tp_orb.debug.txt"
    data_direct_tp_orb = np.array(np.loadtxt(filename_direct_tp_orb, dtype=float))

    alpha = -6.
    beta = -4.
    gamma = -1.
    pt01 = 1.e0
    ID = 30
    data_direct_tp_orb = data_direct_tp_orb[4:-1]
    # index_percent_init  = 0.
    index_percent_init  = 0.1 #0.
    index_percent_final = index_percent_init+0.16

    xv = data_direct_tp_orb[:, 0:6]
    tp = data_direct_tp_orb[:, 6:12]
    tau = tp[:, 0]
    ptau = tp[:, 3]
    lo = len(tau)
    print("len(tau): ", lo)
    # m = np.median(np.abs(ptau))*10
    # wg = np.where(np.abs(ptau)>m)[0]
    # add.DEBUG_PRINT_V(1, wg, len(wg), m)
    # add.DEBUG_PRINT_V(1, ptau)
    ddl = [[xv[:,0:3], "xyz"]]
    PLOT = fff.Plot_model_fit()
    PLOT.plot_x_scatter3d_xxx(ddl)

    pointsize = 0.2
    fontsize = 20.0
    plt.xlabel(r"ellip coor $\lambda$ ($\mathrm{kpc^2}$)", fontsize=fontsize)
    plt.title("particle_ID: %d" % ID, fontsize=fontsize)
    plt.plot([alpha,alpha], [-pt01,pt01], label=r"$\tau=-a^2$", color="orange")
    plt.plot([beta,beta],   [-pt01,pt01], label=r"$\tau=-b^2$", color="orange")
    plt.plot([gamma,gamma], [-pt01,pt01], label=r"$\tau=-c^2$", color="orange")
    plt.plot([min(tau),max(tau)], [0.,0.], label=r"the root line, $y=0$", color="k")

    plt.plot(tau, ptau, label="data", color="b")
    plt.scatter(tau, ptau, label="data", color="b")

    plt.legend(fontsize=fontsize)
    # plt.xscale("log")
    fig_tmp = plt.gcf()
    plt.show()
    print("Fig ... Done.")
    plt.close("all")
