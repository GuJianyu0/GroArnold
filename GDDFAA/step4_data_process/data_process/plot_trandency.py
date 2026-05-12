#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import traceback
import matplotlib.pyplot as plt
import analysis_data_distribution as add
import galaxy_models as gm



if __name__ == "__main__":

    # JT = np.array([ np.logspace(-2., 5., 10000) ]).T
    # # JT = np.array([ np.logspace(-3., 9., 10000) ]).T
    # # JT = np.array([ np.linspace(1e-1, 1e4, 10000) ]).T
    # JO = np.hstack(( JT, JT, JT ))
    # hJ = np.sum(JO[:,0:3], axis=1)
    # add.DEBUG_PRINT_V(1, np.median(hJ))

    folder_name = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/"
    gm_name = "galaxy_general"
    galaxymodel_name = folder_name+gm_name+"/"
    snapshot_ID = 3
    DF_name = "DF_AA_one"
    path_write = galaxymodel_name+"aa/snapshot_%d_%s.txt"%(snapshot_ID, DF_name)
    data_DF = np.loadtxt(path_write)
    tgts = data_DF[:,0:6] #[adjust] fit
    # DF = data_DF[:,6]
    DF = data_DF[:,3]
    # add.DEBUG_PRINT_V(0, data_DF.shape, data_DF[0])
    hJ = np.sum(tgts[:,0:3], axis=1)/1e4
    add.DEBUG_PRINT_V(1, np.percentile(hJ, np.linspace(0., 100., 20)))

    #fitfunc
    p = [9.99999783e+01, 1.70621725e-01, 7.18748061e+01, 1.81400464e-02, 
        1.07440852e+05, 1.07462783e+05, 4.91722159e+04]
    DF_fit = gm.DFAA_fCombinationFree_MDPL_ExpPloy2(tgts, *p)
    add.DEBUG_PRINT_V(1, DF_fit.shape, DF_fit[0])

    # #fitfunc
    # p = [5.e-2,     1., 1.e-1, 1., 2.,  2.e5]
    # DF_fit = gm.DFAA_MDPLE_log(tgts, *p)

    # #fitfunc
    # p = [-1.,       1., 1., 1., 1.,     1.] #reference
    # p = [-1.,       1., 1., 1., 2.,     1.] #n>0, near 2, controls the extremely max position
    # p = [1.e-1,     1., 1., 1., 2.,     1.] #p1>0
    # p = [5.e-2,     1., 1., 1., 2.,     2.e5] #c0 move down and up
    # p = [5.e-2,     1., 1.e-1, 1., 2.,  2.e5] #p3 move up tail
    # p = [5.e-2,     1., 1.e-1, 1., 2.,  2.e5] #p2>0, p4>0 controls the double wings
    # p = [5.e-2,     1., 1.e-1, 1., 2.,  2.e5]
    # DF_fit = gm.DFAA_MDPLE_log(tgts, *p)

    bd_much = [1e0/1e4, 2e4/1e4]
    bd_y = [np.min(DF), np.max(DF)]
    pointsize = 1.
    plt.plot([bd_much[0], bd_much[0]], bd_y, lw=pointsize, color="k")
    plt.plot([bd_much[1], bd_much[1]], bd_y, lw=pointsize, color="k")
    # plt.scatter(hJ, np.log10(DF), s=pointsize)
    # plt.scatter(hJ, DF_fit, s=pointsize)
    plt.scatter(hJ, np.exp(np.log10(DF)), s=pointsize)
    plt.scatter(hJ, np.exp(DF_fit), s=pointsize)
    # plt.ylim(-25., 0.)
    plt.xscale("log")
    # plt.yscale("log")
    plt.show()



    # bd = 1e5
    # N_grid = 10000
    # add.generate_actions_grid_3d_3d(bd, N_grid, tag_func=1)
