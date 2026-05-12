#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import analysis_data_distribution as add
import galaxy_models as gm
import fit_rho_fJ as fff
import affine_trans_group as atg
import triaxialize_galaxy as tg

# main()
if __name__ == '__main__':

    ## expected values
    M = 137.
    N_ptcs = 10000
    rs = 19.6
    ds = 0.004082
    power_Einasto = 1.7
    Js = (gm.G*M*rs)**0.5
    ar_expect = [1., 0.6, 0.3]
    rotateAngle_expected = [0., 0., 0.]

    ##generate a galaxy
    MG = fff.Model_galaxy(M, rs, ds) #do not influence fitting
    id_relative_compare = 1
    coef_boundary = 1.e40
    # coef_boundary = 1.e8
    # scale_boundary = 1.e1
    scale_boundary = 1.e2
    power_boundary = 1.e2
    axisratio_boundary = 1.e2
    MG.set_value("density_scale",   np.array([ds, ds, ds/scale_boundary, ds*scale_boundary]))
    MG.set_value("length_scale",    np.array([rs, rs, rs/scale_boundary, rs*scale_boundary]))
    MG.set_value("rotate_angle_x",  np.array([0., 2*np.pi, 0., 2*np.pi])) # np.pi #divide by zero
    MG.set_value("rotate_angle_y",  np.array([0., 2*np.pi, 0., 2*np.pi]))
    MG.set_value("rotate_angle_z",  np.array([0., 2*np.pi, 0., 2*np.pi]))
    MG.set_value("action_scale",    np.array([Js, Js, Js*1e-1, Js*1e1]))
    MG.set_value("log_penalty",     np.array([-10., -10., -100., 1.]))
    # MG.set_value("coef_total",      np.array([1., 1., 1.-0.1, 1.+0.1]))
    # MG.set_value("coef_total",      np.array([1., 1., 1.e-3, 1.e3]))
    MG.set_value("coef_total",      np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_n3",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_n2",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_n1",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_0",     np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p1",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p2",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p3",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_exp_1",      np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_exp_2",      np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_exp_3",      np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_axis_1",     np.array([1., 1., 1./axisratio_boundary, 1.]))
    MG.set_value("coef_axis_2",     np.array([1., 1., 1./axisratio_boundary, 1.]))
    MG.set_value("coef_axis_3",     np.array([1., 1., 1./axisratio_boundary, 1.]))
    # MG.set_value("power_alpha",     np.array([1., 1., 1e-4,             1e1]))
    # MG.set_value("power_beta",      np.array([3., 3., 2.e-1,            1e2]))
    MG.set_value("power_alpha",     np.array([1., 1., 0.,               power_boundary]))
    MG.set_value("power_beta",      np.array([3., 3., 0.,               power_boundary]))
    MG.set_value("power_total",     np.array([1., 1., 0.,               power_boundary]))
    MG.set_value("power_free_1",    np.array([1., 1., 0.,               power_boundary]))
    MG.set_value("power_free_2",    np.array([1., 1., 0.,               power_boundary]))
    MG.set_value("power_free_3",    np.array([1., 1., 0.,               power_boundary]))
    MG.set_value("power_free_4",    np.array([1., 1., 0.,               power_boundary]))
    MG.set_value("power_Einasto",   np.array([power_Einasto, power_Einasto, power_Einasto/power_boundary, power_Einasto*power_boundary]))

    gm_name = ""
    # gm_name = "_1_NFW_spherical"
    # gm_name = "_4_EinastoUsual_spherical"
    # gm_name = "_11_NFW_triaxial"
    # gm_name = "_41_EinastoUsual_triaxial"
    # snapshot_Id = 0000
    snapshot_Id = 5000



    ####mass density
    whatcannonical = 2 #2: mass density; 5: action density
    method_and_tags = "C2P0S0A0" #mass density
    # method_and_tags = "C5P0S0A0" #SphericalFP; action density, same as below
    # method_and_tags = "C5P0S2A0" #SFFP
    # method_and_tags = "C5P1S2A0" #SFDP
    # method_and_tags = "C5P1S2A1" #PPOD
    # method_and_tags = "C5P1S2A2" #Comb-lmn
    bd = np.inf
    bd_display = bd

    ## each
    galaxybox_name="/home/darkgaia/workroom/0prog/gadget/Gadget-2.0.7/"
    galaxymodel_name = "galaxy_general"+gm_name+"/"
    RD = fff.Read_data_galaxy(MG, gb=galaxybox_name, gmn=galaxymodel_name, wc=whatcannonical) #No such type of data provided
    # doc = RD.data_secondhand_snapshot(snapshot_Id, wc=whatcannonical)
    doc = RD.data_original_NDFA(snapshot_Id, method_and_tags=method_and_tags)
    x, y = RD.data_sample_screen(doc, x_down=1./bd, x_up=bd, is_logy=True)
    xerr = x*0.
    yerr = y*0.1
    x_inp = x[:,0:3]
    v_inp = x[:,3:6]
    add.DEBUG_PRINT_V(1, x_inp[0,0], v_inp[0,0])

    ##triaxializaiton
    basepath = galaxybox_name+galaxymodel_name+"txt/"
    N_ptcs, x_inp1, v_inp1, m = tg.load_snapshot(basepath=basepath, snapshot=snapshot_Id)
    xMEAN, vMEAN = tg.mean_coordinate(x_inp, v_inp, m)
    x_inp, v_inp = tg.centralize_coordinate(x_inp, v_inp, m) #??

    # boundary = [125., 1000.]
    boundary = [60., 1000.]
    IA = tg.interiaMoment_total(x_inp, v_inp, m, boundary=boundary)
    x, v, OA, T, eigenvectors = tg.all_triaxial_process(x_inp, v_inp, m, True, True, boundary=boundary)
    # is_better_triaxial = tg.galaxy_Nbody_triaxialize_comparison(x_inp, v_inp, x, v, is_histogram=True, is_scatter=True, boundary=boundary)

    ##debugs
    # #:: eliminate_totalRotation() donot almost change
    # x1, v1, OA, T = all_triaxial_process(x_inp, v_inp, m, False, True, boundary=boundary)
    # add.DEBUG_PRINT_V(1, mean_coordinate(x1, v1, m), OA, T)
    # is_better_triaxial = galaxy_Nbody_triaxialize_comparison(x1, v1, x, v, is_histogram=True, is_scatter=False, boundary=boundary)
    # #:: rotate_mainAxisDirection() do change, bad
    # x2, v2, OA, T = all_triaxial_process(x_inp, v_inp, m, True, False, boundary=boundary)
    # add.DEBUG_PRINT_V(1, mean_coordinate(x2, v2, m), OA, T)
    # is_better_triaxial = galaxy_Nbody_triaxialize_comparison(x2, v2, x, v, is_histogram=True, is_scatter=False, boundary=boundary)

    # D = np.matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    # theta = 4.304433e+00, 9.737117e-01, 4.775786e+00
    # D1 = atg.SO3(D, theta)
    add.DEBUG_PRINT_V(1, tg.mean_coordinate(x, v, m), IA, OA, eigenvectors, T, "1")
    # x, v, OA, T, eigenvectors = tg.all_triaxial_process(x_inp, v_inp, m, True, True, boundary=boundary)
    # add.DEBUG_PRINT_V(1, tg.mean_coordinate(x, v, m), IA, OA, eigenvectors, T, "2")

    ##data
    is_rewroten = tg.rewrite_snapshot(x, v, m=m, comp_gal=0, basepath=basepath, snapshot=snapshot_Id)

    res = gm.position_estimateScale(x)
    print("AA scale from data: %e" % res)
    # exit(0)

    # ##samples
    # rs = rs
    # vs = np.median(add.norm_l(v, axis=1))
    # x_samples, v_samples = tg.generate_usual_coordinate_samples(rs, vs, 3., 10)
    # m_samples = np.ones(len(x_samples))*m[0]
    # comp_gal = 0
    # is_rewroten = tg.rewrite_snapshot(x_samples, v_samples, m_samples, comp_gal=comp_gal, basepath=basepath, snapshot=snapshot_Id, suffix="samples")
    # add.DEBUG_PRINT_V(1, x_samples.shape, v_samples.shape)



    ####fit and write
    # loop model, data C++/Python, set gal pre, curve fit, compare potential, density_params_fit.txt
    # #A. doublepowerlaw
    # fitmodel = [gm.rho_doublepowerlaw_triaxial_log, gm.rho_doublepowerlaw_triaxial_rotate_log]
    # fitmodel_name = ["gm.rho_doublepowerlaw_triaxial_log", "gm.rho_doublepowerlaw_triaxial_rotate_log"]
    # N_fitmodel = len(fitmodel)
    # fitmodel_params = list(range(N_fitmodel))
    # fitmodel_params_name = [
    #     ["density_scale", "length_scale", "power_alpha", "power_beta", "coef_axis_2", "coef_axis_3", 
    #     "log_penalty"], 
    #     ["density_scale", "length_scale", "power_alpha", "power_beta", "coef_axis_2", "coef_axis_3", 
    #     "rotate_angle_x", "rotate_angle_y", "rotate_angle_z", "log_penalty"]
    # ]

    #B. Sersic
    fitmodel = [gm.rho_Sersic_triaxial_log, gm.rho_Sersic_triaxial_rotate_log]
    fitmodel_name = ["gm.rho_Sersic_triaxial_log", "gm.rho_Sersic_triaxial_rotate_log"]
    N_fitmodel = len(fitmodel)
    fitmodel_params = list(range(N_fitmodel))
    fitmodel_params_name = [
        ["density_scale", "length_scale", "power_Einasto", "coef_axis_2", "coef_axis_3", 
        "log_penalty"], 
        ["density_scale", "length_scale", "power_Einasto", "coef_axis_2", "coef_axis_3", 
        "rotate_angle_x", "rotate_angle_y", "rotate_angle_z", "log_penalty"]
    ]



    #calculation
    for j in np.arange(N_fitmodel):
    # for j in np.arange(1):
        ## 1. fitting
        print(r"fit function %d ..."%(j))
        nf = len(fitmodel_params_name[j])
        WF = fff.Wrap_func(MG, fitmodel[j], nf)
        WF.set_bounds(fitmodel_params_name[j])
        ff = WF.funcfit
        pe = np.array(WF.reference_list)
        CF = fff.Curve_fit_galaxy(WF, nf, x, y, yerr, tag0=0, mf=5000)
        CF.run()
        P_CF, C_CF, R_CF, S_CF, E_CF = CF.display()
        
        # compare potential
        # x0 = np.ones((N_ptcs,3)).astype(float)
        # x0 = x[0] #np.array([-7.12366e+00, -1.87326e+00, -1.00361e+01])
        # pot = gm.potential_nbody_simple(x0, x, M/N_ptcs, soft=5.)
        # print(pot)
        # write params value; re-run the two model and iterate??
        density_params_fit = "/home/darkgaia/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general/"
        density_params_fit += "density_params_fit_%03d_%s.txt"%(snapshot_Id, fitmodel_name[j])
        file_handle = open(density_params_fit, mode="w")
        file_handle.write("##fit paramters of mass density\n")
        for k in np.arange(0,2): #ds rs qy qz p1 p2
            file_handle.write("%e #%s\n"%(P_CF[k], fitmodel_params_name[j][k]))
        pw = np.zeros(2)
        pw[0] = P_CF[2] #for Einasto, power_Einasto ~ index_2
        pw[1] = 0. #for Einasto, no such power paramter
        for k in np.arange(nf-3, nf-1):
            file_handle.write("%e #%s\n"%(P_CF[k], fitmodel_params_name[j][k]))
        for k in np.arange(0,2): #ds rs qy qz p1 p2
            file_handle.write("%e #%s\n"%(pw[k], "power%d"%(k)))
        file_handle.write("%e %e %e %e %e %e #%s\n"%(xMEAN[0], xMEAN[1], xMEAN[2], 
            vMEAN[0], vMEAN[1], vMEAN[2], "to centralization"))
        file_handle.write("%e %e %e #%s\n"%(OA[0], OA[1], OA[2], 
            "to fix total rotation by frequencies by interiaMoment"))
        file_handle.write("%e %e %e %e %e %e %e %e %e #%s\n"%(T[0,0], T[0,1], T[0,2], 
            T[1,0], T[1,1], T[1,2], T[2,0], T[2,1], T[2,2], 
            "to fix direction by main axis of total interia moment"))
        # for k in np.arange(nf-1):
        #     file_handle.write("%e #%s\n"%(P_CF[k], fitmodel_params_name[j][k]))
        # file_handle.write("%e %e %e #%s\n"%(0., 0., 0., "rotateAngle_x ...y ...z"))
        file_handle.close()
        print("Wrote done.")



        # ## potential difference from expect potential #however, I donot konw
        # if j==0:
        #     n1 = 3
        #     n2 = 3
        #     rsrs = 19.6
        #     X,Y,Z = add.generate_sample_square(n1,n2,rsrs)

        #     PD = np.zeros((n1,n2))
        #     PF = np.zeros((n1,n2))
        #     for i in np.arange(n1):
        #         for j in np.arange(n2):
        #             target = np.array([X[i,j], Y[i,j], Z[i,j]])
        #             PD[i,j] = gm.potential_nbody_simple(target, x)
        #             # PF[i,j] = gm.Phi_doublepowerlaw_triaxial(target, *P_CF)
        #             PF[i,j] = gm.Phi_doublepowerlaw_triaxial(target, ds,rs,1.,3.,0.6,0.3)
        #             # print("pot calcu: ", i,j)
        #     PD = np.log(-PD)/np.log(10.)
        #     PF = np.log(-PF)/np.log(10.)
        #     print(PD, PF)
        #     # add.DEBUG_PRINT_V(1, X,Y,Z, PD,PF,"grid")
        #     ##many sanpshots??

        #     ##plot
        #     pointsize = 0.2
        #     fontsize = 6.0
        #     dpi = 500
        #     fig = plt.figure(dpi=dpi)
        #     # fig = plt.figure(dpi=300, figsize=(10,2)) #show, not axis length
        #     ax=fig.add_subplot(1,1,1, projection='3d')
        #     ax.grid(True)

        #     # cm = plt.cm.get_cmap('gist_rainbow') #rainbow
        #     # axsc = ax.scatter(Xl[:,0], Yl[:,1], Zl[:,2], s=pointsize, label="", c=Pl, cmap=cm)
        #     # plt.colorbar(axsc)

        #     surf = ax.plot_surface(X, Y, PD, cmap=cm.jet, linewidth=0, antialiased=False, label="potential by nbody data")
        #     surf = ax.plot_surface(X, Y, PF, cmap=cm.jet, linewidth=0, antialiased=False, label="potential by fit formula")
        #     position=fig.add_axes([0.1, 0.3, 0.02, 0.5]) #x pos, ypos, width, shrink
        #     # fig.colorbar(surf, cax=position, aspect=5)

        #     ax.set_xlabel(r"x", fontsize=fontsize)
        #     ax.set_ylabel(r"y", fontsize=fontsize)
        #     ax.set_zlabel(r"z", fontsize=fontsize)
        #     ax.set_zlabel(r"potential", fontsize=fontsize)
        #     # ax.set_title(r"potential", fontsize=fontsize)
        #     # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)
        #     plt.show()

        # # sample E surface; sample xv~J with bg; sample stream
        # aaaa = 1

    # ##display running time
    # import time
    # t1 = time.perf_counter()
    # for i in range(10000):
    #     aaaa = i*i
    # t2 = time.perf_counter()
    # print(t1,t2)
