#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as spopt

import analysis_data_distribution as add
import galaxy_models as gm
# import fit_rho_fJ as fff
import RW_data_CMGD as rdc
import KDTree_python as kdtp
import plot_galaxy_wrapper as pgw



if __name__ == '__main__':

    #### A: 
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
    # snapshot_Id = 5000 #bad 6000 exp pot
    # snapshot_Id = 80
    snapshot_Id = 160

    # bd = 1e5/2
    # bd = 2e5
    # bd = 1e4
    bd = 1e6
    # bd = 2e6
    bd_min = 1e-2
    bd_display = bd
    filename = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/aa/"\
        +"snapshot_160.action.method_all.txt"
    
    xv, AA_cl, mass = rdc.read_actions(filename, bd=bd, bd_min=bd_min, is_angles=False, actionmethod="AA_TF_DP")
    Jl_m = np.median(AA_cl[:,0]) #big
    Jm_m = np.median(AA_cl[:,1])
    Jn_m = np.median(AA_cl[:,2])
    fJ_m = 1.

    # xv, AA_cl, mass = rdc.read_actions(filename, bd=bd, is_angles=False, actionmethod="AA_TF_DP")
    add.DEBUG_PRINT_V(1, len(AA_cl), "len(AA_cl)")

    cols = [0,1,2]
    KD = kdtp.KDTree_galaxy_particles(AA_cl[:,cols], weight_extern_instinct=mass)
    DF = KD.density_SPH(AA_cl[:,cols]) #some are None #?? debug
    DF_log10 = np.log10(DF)
    add.DEBUG_PRINT_V(1, len(DF_log10), "len(DF_log10)")
    
    Jl = AA_cl[:,0]
    Jm = AA_cl[:,1]
    Jn = AA_cl[:,2]
    fJ = DF_log10
    Jl_frac_m = Jl/Jl_m
    Jm_frac_m = Jm/Jm_m
    Jn_frac_m = Jn/Jn_m
    fJ_frac_m = fJ/fJ_m
    add.DEBUG_PRINT_V(1, Jl_m, Jm_m, Jn_m, fJ_m, "medians")

    PLOT = pgw.Plot_model_fit()
    dim = 3
    plot_list = [
        # [Jl_frac_m, Jm_frac_m, Jn_frac_m]
        # [Jl, Jm, Jn]
        [np.log10(Jl), np.log10(Jm), np.log10(Jn)]
    ]
    f_list = [fJ]
    label_list = [
        "action_space_points"
    ]
    median_rate = 3.
    # median_rate = 10.
    xyzlim = [
        # [0., median_rate], #rate of median
        # [0., median_rate], 
        # [0., median_rate]
        [np.log10(bd_min), np.log10(bd)], #log10
        [np.log10(bd_min), np.log10(bd)], 
        [np.log10(bd_min), np.log10(bd)]
        # [0., np.log10(bd)], #log10
        # [0., np.log10(bd)], 
        # [0., np.log10(bd)]
        # None, 
        # None, 
        # None
        # [0.1, 10000.], #abs
        # [0.1, 6000.], 
        # [0.1, 3000.]
        # [0.1, bd/100.], 
        # [0.1, bd/200.], 
        # [0.1, bd/200.]
    ]
    # scalevalue = [
    #     Js, 
    #     Js, 
    #     Js
    # ]
    xyztitle = [
        # "J_lambda/median (unit 1)", 
        # "J_mu/median (unit 1)", 
        # "J_nv/median (unit 1)"
        # "J_lambda (kpc*km/s)", 
        # "J_mu (kpc*km/s)", 
        # "J_nv (kpc*km/s)"
        # r"log10(${J_\lambda}) (log10 kpc*km/s)", 
        # r"log10(${J_\mu}) (log10 kpc*km/s)", 
        # r"log10(${J_\nv}) (log10 kpc*km/s)"
        r"log10(${J_\lambda}$) (log10 kpc*km/s)", 
        r"log10(${J_\mu}$) (log10 kpc*km/s)", 
        r"log10(${J_\nu}$) (log10 kpc*km/s)"
    ]
    view_angles = [(1.5)*np.pi, (0.1)*np.pi]
    PLOT.plot_scatter_2d_or_3d_with_color(plot_list, f_list=f_list, 
        label_list=label_list, dim=dim, xyzlim=xyzlim, xyztitle=xyztitle, 
        view_angles=view_angles
    )



    # ####slices
    # yT = np.array([y]).T
    # y0, dy0 = -1.e0, 1.e2 #-18. ~ -8. #for all PODD
    # # y0, dy0 = 10., 0.1
    # # y0, dy0 = 10.2, 0.1 #bad
    # # y0, dy0 = 10.5, 0.1 #bad
    # # y0, dy0 = 11., 0.1 #bad
    # # y0, dy0 = 11.5, 0.1
    # # y0, dy0 = 12., 0.03 #bad
    # # y0, dy0 = 12., 0.1 #bad
    # # y0, dy0 = 13., 0.1
    # # y0, dy0 = 13.5, 0.1
    # # y0, dy0 = 14., 0.1
    # # y0, dy0 = 15., 0.1
    # y_dy, cl, cnl = add.screen_boundary(yT, y0, y0+dy0)
    # add.DEBUG_PRINT_V(1, cl, len(cl))
    # xlog = np.log(x)
    # x_dy = x[cl]
    # xlog_dy = xlog[cl]

    # x = x #J[3]
    # yT = np.array([y]).T #log_fJ
    # ymin = -16.
    # ymax = -10.
    # dy = 0.2
    # N_slices = int((ymax-ymin)/dy)+1
    # DY = np.linspace(-16., -10., N_slices)
    # # add.DEBUG_PRINT_V(0, DY, len(DY))
    # N_sample_min = 4
    # params = np.zeros((len(DY)-1, 3)) #params

    # for i in np.arange(len(DY)-1):
    #     y0 = DY[i]
    #     dy0 = DY[i+1]-DY[i]
    #     absmin = DY[i]
    #     absmax = DY[i+1]
    #     # add.DEBUG_PRINT_V(1, absmin, absmax)

    #     yT_dy, cl, cnl = add.screen_boundary_PM(yT, absmin, absmax)
    #     add.DEBUG_PRINT_V(1, cl, len(cl))
    #     x_dy = x[cl, 0:3]
    #     y_dy = y[cl]
    #     # xlog = np.log(x)
    #     # xlog_dy = xlog[cl]
    #     # y_err = 0.*y_dy
    #     # add.DEBUG_PRINT_V(1, x_dy, y_dy, "xy")
    #     slope_ref = 10.
    #     scale_ref = -np.mean(y_dy/np.sum(x_dy, axis=1))
    #     # add.DEBUG_PRINT_V(1, y_dy.shape, np.sum(x_dy, axis=1), "xyshape")
    #     add.DEBUG_PRINT_V(1, scale_ref, "scale_ref")

    #     if len(cl)<N_sample_min: #too less point
    #         params[i] = np.array([scale_ref*1., -scale_ref*1., -scale_ref*1.]) #bad value
    #     else:
    #         funcfit = gm.surface_plane
    #         # funcfit = gm.surface_plane #curvature??
    #         p0 = [scale_ref*1., scale_ref*1., scale_ref*1.]
    #         boundsD = [scale_ref/slope_ref, scale_ref/slope_ref, scale_ref/slope_ref]
    #         boundsU = [scale_ref*slope_ref, scale_ref*slope_ref, scale_ref*slope_ref]
    #         optimization, covariance = spopt.curve_fit(funcfit, x_dy, y_dy, 
    #             p0 = p0, bounds = (boundsD, boundsU), maxfev = 5000)
    #         params[i] = np.array(optimization)
    #     print(i, params[i])

    # slope_YX = params[:,1]/params[:,0] #slopes of Y-X and Z-X
    # slope_ZX = params[:,2]/params[:,0]
    # DYL = np.abs(DY[:-1])
    
    # ####plot
    # pointsize = 10.
    # fontsize = 20.
    # plt.scatter(DYL, slope_YX, s=pointsize, color="red",  label="J1 to J0")
    # plt.scatter(DYL, slope_ZX, s=pointsize, color="blue", label="J2 to J0")
    # plt.plot([np.min(DYL), np.max(DYL)], [0., 0.], color="black")
    # plt.plot([np.min(DYL), np.max(DYL)], [1., 1.], color="green")
    # plt.xlabel(r"abs logrithm of numerical of dtribution function of actions, $\|\log(N(J012))\|$", fontsize=fontsize)
    # plt.ylabel(r"actions slopes of J1 to J0 or J2 to J0, k", fontsize=fontsize)
    # plt.legend(fontsize=fontsize)
    # plt.show()
