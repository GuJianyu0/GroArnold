#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from scipy.interpolate import RBFInterpolator
import KDTree_python as kdtp
import analysis_data_distribution as add
import galaxy_models as gm
import RW_data_CMGD as rdc
import triaxialize_galaxy as tg

def frac_to_median(A_2d): #module add.?? axis??
    n, m = np.shape(A_2d)
    for i in np.arange(m):
        A_2d[:,i] /= np.median(A_2d[:,i])
    return A_2d

if __name__ == "__main__":

    #### [split code]
    # #### prepare data
    # filename = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/"\
    #     "galaxy_general/aa/snapshot_80.action.method_all.txt" #80
    # bd = 1e6
    # is_angles = True
    # cols = [0,1,2]
    # xv, AA_cl, mass = rdc.read_actions(filename, bd=bd, is_angles=is_angles)
    # x = xv[:,0:3]
    # v = xv[:,3:6]
    # rx = add.norm_l(x, axis=1)
    # rv = add.norm_l(v, axis=1)
    # L = tg.angularMoment(x, v)
    # La = np.abs(L)

    # # add.DEBUG_PRINT_V(1, np.shape(AA_cl), np.median(rx), np.median(rv), "xv_meds")
    # # for i in np.arange(3):
    # #     add.DEBUG_PRINT_V(1, np.median(La[:,i]), np.median(AA_cl[:,i]), 
    # #         np.median(AA_cl[:,i+3]), "%d: L, AA"%(i))
    # # add.DEBUG_PRINT_V(1, np.median(np.sum(La, axis=1)), np.median(np.sum(AA_cl[:,0:3], axis=1)), 
    # #     np.median(np.sum(AA_cl[:,3:6], axis=1)), "sum: L, AA")
    
    # add.DEBUG_PRINT_V(1, np.shape(AA_cl), np.mean(rx), np.mean(rv), "xv_means")
    # for i in np.arange(3):
    #     add.DEBUG_PRINT_V(1, np.mean(La[:,i]), np.mean(AA_cl[:,i]), 
    #         np.mean(AA_cl[:,i+3]), "%d: L, AA"%(i))
    # add.DEBUG_PRINT_V(1, np.mean(np.sum(La, axis=1)), np.mean(np.sum(AA_cl[:,0:3], axis=1)), 
    #     np.mean(np.sum(AA_cl[:,3:6], axis=1)), "sum: L, AA")



    #### [split code]
    # #### prepare data
    filename = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/"\
        "galaxy_general_DPL_less_20221107/aa/snapshot_40.action.method_all.txt" #80
    bd = 1e6
    is_angles = True
    cols = [0,1,2]
    xv, AA_cl, mass = rdc.read_actions(filename, bd=bd, is_angles=is_angles)
    xv = frac_to_median(xv) #deep copy??
    # AA_cl = frac_to_median(AA_cl)
    mass = mass/mass
    x = xv[:,0:3]
    v = xv[:,3:6]
    J = AA_cl[:, 0:3]
    F = AA_cl[:, 3:6]
    O = AA_cl[:, 6:9]
    JO = np.hstack((J,O))
    means = add.averages_by_angle_action_data(AA_cl)
    add.DEBUG_PRINT_V(1, means, "meds")

    #### OJ to xv in known xv0 data: knn interp, data xv0, input OJ, output xv, use medianed, 1d
    JO = np.random.random((100,6))
    xv = JO**2
    mass = np.random.random(100)

    tgts = np.random.random((1000000,6))
    print(tgts)
    tgts_varational = tgts*( 1. + 0.01*np.random.random(np.shape(tgts)) )
    print(tgts_varational)
    # add.DEBUG_PRINT_V(1, np.shape(JO), np.shape(tgts), "tgts")

    KD = kdtp.KDTree_galaxy_particles(JO, weight_extern_instinct=mass)
    distances, indices = KD.query(tgts)
    # add.DEBUG_PRINT_V(1, indices, "indices")
    for i in np.arange(len(tgts)):
        # add.DEBUG_PRINT_V(1, np.shape(indices[i]), np.shape(indices), "ii")
        rbffunc = RBFInterpolator(JO[indices[i]], xv[indices[i],:], neighbors=32, kernel="thin_plate_spline")
        fitintpy = rbffunc([tgts_varational[i]])
        add.DEBUG_PRINT_V(1, i, xv[indices[i][0],:], fitintpy[0], "xf")



    #### [split code]
    # import matplotlib.pyplot as plt
    # from scipy.interpolate import RBFInterpolator
    # from scipy.stats.qmc import Halton

    # rng = np.random.default_rng()
    # xobs = 2*Halton(2, seed=rng).random(100) - 1
    # yobs = np.sum(xobs, axis=1)*np.exp(-6*np.sum(xobs**2, axis=1))

    # xgrid = np.mgrid[-1:1:50j, -1:1:50j]
    # print(xgrid.shape)
    # xflat = xgrid.reshape(2, -1).T
    # yflat = RBFInterpolator(xobs, yobs)(xflat)
    # ygrid = yflat.reshape(50, 50)

    # fig, ax = plt.subplots()
    # ax.pcolormesh(*xgrid, ygrid, vmin=-0.25, vmax=0.25, shading='gouraud')
    # p = ax.scatter(*xobs.T, c=yobs, s=50, ec='k', vmin=-0.25, vmax=0.25)
    # fig.colorbar(p)
    # plt.show()
