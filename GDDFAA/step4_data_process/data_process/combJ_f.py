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

    #### A: 
    ## usual settings
    M = 137.
    ls = 19.6
    ds = 0.004568 #0.000891
    Js = (gm.G*M*ls)**0.5
    id_relative_compare = 1

    gm_name = ""
    # gm_name = "_A"
    # gm_name = "_B"
    # gm_name = "_1_NFW_spherical"
    # gm_name = "_4_EinastoUsual_spherical"
    # gm_name = "_11_NFW_triaxial"
    # gm_name = "_41_EinastoUsual_triaxial"
    galaxymodel_name = "galaxy_general"+gm_name+"/"
    # snapshot_Id = 0000 #sometimes well 8000 exp pot
    snapshot_Id = 5000 #bad 6000 exp pot

    whatcannonical = 5
    # method_and_tags = "C5P0S0A0" #SphFP
    # method_and_tags = "C5P0S2A0" #SFFP
    method_and_tags = "C5P1S2A0" #SFDP
    # method_and_tags = "C5P1S2A1" #TEPPODI
    # method_and_tags = "C5P1S2A2" #Comb-lmn

    bd = 1e5
    # bd = 1e5/2
    # bd = 2e6
    bd_display = bd

    MG = fff.Model_galaxy(M, ls, ds)
    RD = fff.Read_data_galaxy(MG, gmn=galaxymodel_name, wc=whatcannonical)
    doc = RD.data_original_NDFA(snapshot_Id, method_and_tags=method_and_tags)
    x, y = RD.data_sample_screen(doc, x_down=1./bd, x_up=bd, is_logy=True, is_abs=True)
    xerr = x*0.1
    yerr = y*0.1
    x = np.hstack((x[:, 3:6], x[:, 0:3]))
    Jes = gm.AA_combination_estimateScale(x)
    print("AA scale from data: %e" % Jes)

    yT = np.array([y]).T
    y0, dy0 = -1.e0, 1.e2 #-18. ~ -8. #for all PODD
    # y0, dy0 = 10., 0.1
    # y0, dy0 = 10.2, 0.1 #bad
    # y0, dy0 = 10.5, 0.1 #bad
    # y0, dy0 = 11., 0.1 #bad
    # y0, dy0 = 11.5, 0.1
    # y0, dy0 = 12., 0.03 #bad
    # y0, dy0 = 12., 0.1 #bad
    # y0, dy0 = 13., 0.1
    # y0, dy0 = 13.5, 0.1
    # y0, dy0 = 14., 0.1
    # y0, dy0 = 15., 0.1
    y_dy, cl, cnl = add.screen_boundary(yT, y0, y0+dy0)
    add.DEBUG_PRINT_V(1, cl, len(cl))
    xlog = np.log(x)
    x_dy = x[cl]
    xlog_dy = xlog[cl]

    # pointsize = 0.2
    # fontsize = 20.0
    # plt.plot(x,y)
    # plt.xlabel(r"ellip coor $\lambda$ ($\mathrm{kpc^2}$)", fontsize=fontsize)
    # plt.ylabel(r"ellip momentum root solve function in Fudge at $\lambda$ ($kpc^2$)", fontsize=fontsize)
    PLOT = fff.Plot_model_fit()
    ddl = [ [x_dy[:, 0:3], y_dy, "%s: (log(NDF(J))) at %f~-%f"%(method_and_tags, -(y0+dy0), y0)] ]
    PLOT.plot_x_scatter3d_dd(ddl, bd=bd_display, is_show=1)
    # PLOT.plot_x_scatter3d_dd(ddl, is_lim=True, bd=bd_display, is_show=1)



    # #### A: 
    # # filename = "/home/darkgaia/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/snapshot_200.action.method_directorbit.txt"
    # filename = "/home/darkgaia/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/snapshot_500.action.method_all.txt"
    # data = np.array(np.loadtxt(filename, dtype=float))

    # # J = data[:, 63:66]
    # # J = data[:, -8:-5]
    # # J = data[:, -4:-1]
    # # column_J = [-4,64,65]
    # column_J = [-4,63, 11]
    # # column_J = [-3,64, 11]
    # # column_J = [-2,65, 11]
    # J = data[:, column_J]
    # bd = 5.e4

    # fig = plt.figure()
    # pointsize = 0.5
    # fontsize = 10.0
    # ax = fig.add_subplot(1,1,1, projection='3d')
    # ax.scatter(J[:,0], J[:,1], J[:,2], label=r"actions J1J2J3", s=pointsize)

    # ax.legend(fontsize=fontsize)
    # ax.set_xlabel(r"x", fontsize=fontsize)
    # ax.set_ylabel(r"y", fontsize=fontsize)
    # ax.set_zlabel(r"z", fontsize=fontsize)
    # ax.set_xlim(0., bd)
    # ax.set_ylim(0., bd)
    # ax.set_zlim(0., bd)
    # ax.set_ylim(0., bd/2)
    # ax.set_zlim(0., bd/2)

    # whspace = 0.4
    # plt.subplots_adjust(hspace=whspace, wspace=whspace)
    # plt.suptitle(r"")
    # plt.show()
    # plt.close()
    