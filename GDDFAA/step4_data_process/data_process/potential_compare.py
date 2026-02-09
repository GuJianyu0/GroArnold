#!/usr/bin/env python
# -*- coding:utf-8 -*-

from logging import raiseExceptions
from math import radians
from secrets import randbits
from time import gmtime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import galaxy_models as gm
import analysis_data_distribution as add



if __name__ == '__main__':

    ## read data
    fname = "./data_tmp/potential_forces_compare.txt"
    data = np.array(np.loadtxt(fname))
    N_total = len(data)
    data = data[0:1000]
    N_display = len(data)
    x = data[:,0:6]
    r = add.norm_l(x, axis=1)
    ID = data[:,6]
    L = len(ID)

    #:: each row: pot, F, pot_xx, nouse #see label
    P_gadget = data[:, 7]
    P_sum = data[:, 8]
    P_sump = data[:, 9]
    P_sum1 = data[:, 10]
    P_sump1 = data[:, 11]

    F_gadget = data[:,12:15]
    F_sum = data[:,15:18]
    F_sum1 = data[:,18:21]

    l = 1
    err_sph = add.norm_l((P_sum-P_gadget), l=l)/len(r)
    err_sphp = add.norm_l((P_sump-P_gadget), l=l)/len(r)
    err_plu = add.norm_l((P_sum1-P_gadget), l=l)/len(r)
    err_plup = add.norm_l((P_sump1-P_gadget), l=l)/len(r)
    add.DEBUG_PRINT_V(1, err_sph, err_sphp, err_plu, err_plup, "P_err")

    err_sphf = add.norm_l((F_sum[:,0]-F_gadget[:,0]), l=l)/len(r)
    err_pluf = add.norm_l((F_sum1[:,0]-F_gadget[:,0]), l=l)/len(r)
    add.DEBUG_PRINT_V(1, err_sphf, err_pluf, "F_err")

    ## plot x00-P
    szpp = 10.
    szft = 10.
    label = [r"gadget", r"sum", r"sump", r"sum1", r"sump1"]
    color = ["k", "r", "orange", "g", "b", "purple"]
    marker = ["+", "v", "o"]

    plt.subplot(3,2,1)
    plt.scatter(r, P_gadget,    s=szpp, label=label[0], color=color[0], marker=marker[0])
    plt.scatter(r, P_sum,       s=szpp, label=label[1], color=color[1], marker=marker[0])
    plt.scatter(r, P_sump,      s=szpp, label=label[2], color=color[2], marker=marker[0])
    plt.scatter(r, P_sum1,      s=szpp, label=label[3], color=color[3], marker=marker[0])
    plt.scatter(r, P_sump1,     s=szpp, label=label[4], color=color[4], marker=marker[0])
    plt.xlabel(r'$x\quad \mathrm{kpc}$', fontsize=szft)
    plt.ylabel(r'potential on x-axis', fontsize=szft)
    plt.legend(fontsize=szft)

    plt.subplot(3,2,2)

    for i in np.arange(3):
        plt.subplot(3,2,i+3)
        plt.scatter(r, F_gadget[:,i],    s=szpp, label=label[0], color=color[0], marker=marker[0])
        plt.scatter(r, F_sum[:,i],       s=szpp, label=label[1], color=color[1], marker=marker[0])
        plt.scatter(r, F_sum1[:,i],      s=szpp, label=label[3], color=color[3], marker=marker[0])
        plt.xlabel(r'$x\quad \mathrm{kpc}$', fontsize=szft)
        plt.ylabel(r'Forces component on x-axis', fontsize=szft)
        plt.legend(fontsize=szft)

    plt.subplot(3,2,6)
    plt.scatter(r, add.norm_l(F_gadget, axis=1, l=2), s=szpp, label=label[0], color=color[0], marker=marker[0])
    plt.scatter(r, add.norm_l(F_sum, axis=1, l=2),    s=szpp, label=label[1], color=color[1], marker=marker[0])
    plt.scatter(r, add.norm_l(F_sum1, axis=1, l=2),   s=szpp, label=label[3], color=color[3], marker=marker[0])
    plt.xlabel(r'$x\quad \mathrm{kpc}$', fontsize=szft)
    plt.ylabel(r'Forces norm on x-axis', fontsize=szft)
    plt.legend(fontsize=szft)

    plt.suptitle("comparation; display %d/%d of particles" %(N_display, N_total))
    plt.show()



    ####SCF
    # ## read data
    # fname = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/snapshot_000.potentialcompare.txt"
    # data = np.array(np.loadtxt(fname))

    # data_finite = np.array(np.where(np.isfinite(data), data,0.)) #remove not finite value
    # data = abs(data_finite) #not minus
    
    # ID = data[:,0]
    # x0 = data[:,1]
    # print(ID,x0)
    # L = len(ID)

    # #:: each row: pot, F, pot_xx, nouse #see label
    # P0 = data[:, 2]  
    # P1 = data[:, 3] 
    # P2 = data[:, 4]
    # P3 = data[:, 5]

    # ## plot x00-P
    # szpp = 2.
    # szft = 10.
    # label = [r"DS", r"SCF", r"SPH", r"RBF"]
    # color = ["k", "r", "b", "g"]
    # marker = ["o", "x", "v", "^"]

    # # plt.subplot(2,2,1)
    # plt.scatter(x0, P0, s=szpp, label=label[0], color=color[0], marker=marker[0])
    # plt.scatter(x0, P1, s=szpp, label=label[1], color=color[1], marker=marker[1])
    # plt.scatter(x0, P2, s=szpp, label=label[2], color=color[2], marker=marker[2])
    # plt.scatter(x0, P3, s=szpp, label=label[3], color=color[3], marker=marker[3])
    # plt.xlabel(r'$x\quad \mathrm{kpc}$', fontsize=szft)
    # plt.ylabel(r'potential in a line {x, 2., 1.}', fontsize=szft)

    # plt.xscale("log")
    # plt.yscale("log")
    # plt.legend(fontsize=szft)
    # plt.show()



    ####dx dxx dxxx
    # ## read data
    # fname = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/snapshot_000.potentialcompare.txt"
    # data = np.array(np.loadtxt(fname))
    # # D_num = np.where(data==np.inf, 0., D)
    
    # x0 = data[:,0]
    # ID = data[:,1]
    # L = len(ID)

    # #:: each row: pot, F, pot_xx, nouse #see label
    # P0 = data[:, 3:11]   
    # P1 = data[:, 11:19]  
    # P2 = data[:, 19:27] 
    # P3 = data[:, 27:35] 

    # ## plot x00-P
    # szpp = 10.
    # szft = 10
    # label = [r"formula", r"SCF diff", r"sum sum", r"sum diff"]
    # color = ["k", "r", "b", "g"]
    # marker = ["^", "v", "o", "x"]

    # plt.subplot(2,2,1)
    # plt.scatter(x0, P0[:,0], s=szpp, label=label[0], color=color[0], marker=marker[0])
    # plt.scatter(x0, P1[:,0], s=szpp, label=label[1], color=color[1], marker=marker[1])
    # plt.scatter(x0, P2[:,0], s=szpp, label=label[2], color=color[2], marker=marker[2])
    # plt.scatter(x0, P3[:,0], s=szpp, label=label[3], color=color[3], marker=marker[3])
    # plt.xlabel(r'$x\quad \mathrm{kpc}$', fontsize=szft)
    # plt.ylabel(r'potential on x-axis {x, 0., 0.}', fontsize=szft)
    # plt.legend(fontsize=szft)

    # plt.subplot(2,2,2)
    # plt.scatter(x0, P0[:,1], s=szpp, label=label[0], color=color[0], marker=marker[0])
    # plt.scatter(x0, P1[:,1], s=szpp, label=label[1], color=color[1], marker=marker[1])
    # plt.scatter(x0, P2[:,1], s=szpp, label=label[2], color=color[2], marker=marker[2])
    # plt.scatter(x0, P3[:,1], s=szpp, label=label[3], color=color[3], marker=marker[3])
    # plt.xlabel(r'$x\quad \mathrm{kpc}$', fontsize=20)
    # plt.ylabel(r'potentialx on x-axis {x, 0., 0.}', fontsize=szft)
    # plt.ylim(-10000.,1000.)
    # plt.legend(fontsize=szft)

    # plt.subplot(2,2,3)
    # plt.scatter(x0, P0[:,4], s=szpp, label=label[0], color=color[0], marker=marker[0])
    # plt.scatter(x0, P1[:,4], s=szpp, label=label[1], color=color[1], marker=marker[1])
    # plt.scatter(x0, P2[:,4], s=szpp, label=label[2], color=color[2], marker=marker[2])
    # plt.scatter(x0, P3[:,4], s=szpp, label=label[3], color=color[3], marker=marker[3])
    # plt.xlabel(r'$x\quad \mathrm{kpc}$', fontsize=20)
    # plt.ylabel(r'potentialxx on x-axis {x, 0., 0.}', fontsize=szft)
    # plt.ylim(-1000.,1000.)
    # plt.legend(fontsize=szft)

    # # plt.title(r"comparing")
    # # plt.savefig(fname+"_potential_compare.png", dpi=200)
    # plt.show()
