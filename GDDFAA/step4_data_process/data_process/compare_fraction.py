#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sys import argv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import analysis_data_distribution as add
import triaxialize_galaxy as tg
# import galaxy_models as gm
import fit_rho_fJ as fff
# import integrate_methods as im



Dim = 3
colors = ["red", "orange", "olive", "green", "cyan", "blue", "purple", "pink", "gray", "black"]
# colors = ["red", "orange", "gold", "green", "cyan", "blue", "purple"]

def compare_scatter_line(P, N_subplot, N_curve, name_subplot=None, name_curve=None):
    pointsize = 1.
    fontsize = 9.
    for j in np.arange(N_subplot):
        plt.subplot(3,2,j+1)
        for i in np.arange(N_curve):
            x_ploted = P[0] #??
            y_ploted = P[i][:,j]
            # y_ploted = np.abs(P[i][:,j])
            y_ploted = add.log_abs_P1(P[i][:,j], logbasis=10.)
            # add.DEBUG_PRINT_V(0, (P[i][:,j])[0], y_ploted[0])
            
            plt.plot([np.min(x_ploted), np.max(x_ploted)], [0., 0.], lw=pointsize, color="black")
            plt.scatter(x_ploted, y_ploted, s=pointsize, color=colors[i])
            lb = None
            if name_curve!=None:
                lb = "%s"%(name_curve[i])
            plt.plot(x_ploted, y_ploted, label=lb, lw=pointsize, color=colors[i])
        rb = 5e2
        pb = 2e5
        # plt.xlim(1./rb, rb)
        # if j==0:
        #     plt.ylim(-1./pb, pb*2.)
        # else:
        #     plt.ylim(-pb, pb)
        plt.xscale("log")
        # plt.yscale("log")
        nm = None
        if name_subplot!=None:
            nm = "%s"%(name_subplot[j])
        plt.title(nm)
        if j==0:
            plt.legend(fontsize=fontsize)
            legend = plt.legend()
            frame = legend.get_frame()
            frame.set_alpha(1)
            frame.set_facecolor('none') 

    soft = 0.5
    plt.suptitle(r"The potential and forces comparasion, softening %.2ekpc.""\n"%(soft)\
        +r"x value: A stright line along vector \{1,1,1\}; units \{1e10M_{sun},kpc,km/s\}, similarly hereafter.""\n"\
        +r"y value: h(g): sign(g)*log10(|g|+1), the meaning of g is to see each subtitle.""\n")
    plt.show()
    plt.close()

def compare_fraction(XX, VB, XX_name, VB_name, N_xvar, N_yvar, N_curve, fracbound=1e2):
    dpi = 300
    pointsize = 0.2
    fontsize = 1.6

    fig = plt.figure(dpi=dpi)
    for i_xvar in np.arange(N_xvar):
        for swit in np.arange(N_yvar):
            add.DEBUG_PRINT_V(1, N_xvar, N_yvar, N_xvar*swit+i_xvar+1)
            # ax = fig.add_subplot(Dim+1, (swit%2)+1, int(np.ceil(swit/2))+1)
            ax = fig.add_subplot(N_yvar, N_xvar, N_xvar*swit+i_xvar+1)
            xx = XX[i_xvar]

            for mtd in np.arange(N_curve):
                vb = VB[1][mtd]/VB[0][mtd]
                if len(vb.shape)>1:
                    vb = vb[:, swit]
                label = r"Ellip coor %d, %s ~ $\log$(Value_[%s]/Value_[%s])"\
                    %(swit, XX_name[i_xvar], VB_name[1][mtd],VB_name[0][mtd])
                ax.scatter(xx, vb, s=pointsize, label=label, color=colors[mtd], marker="x", alpha=0.8)
            
            min_xvar = min(xx)
            max_xvar = max(xx)
            stdline = 1.
            ax.plot([min_xvar, max_xvar], [stdline,stdline], color="black", lw=pointsize)
            stdline = 0.5
            ax.plot([min_xvar, max_xvar], [stdline,stdline], color="black", lw=pointsize)
            stdline = 2.
            ax.plot([min_xvar, max_xvar], [stdline,stdline], color="black", lw=pointsize)
            
            ax.set_xscale("log")
            ax.set_yscale("log")
            if fracbound>0.:
                # ax.set_xlim()
                ax.set_ylim(1./fracbound, fracbound)
            plt.legend(fontsize=fontsize)
            plt.tick_params(labelsize=fontsize)

    plt.suptitle("%s ~ %s"%("", "")
        , fontsize=fontsize)
    whspace = 0.5
    plt.subplots_adjust(hspace=whspace, wspace=whspace)
    plt.show()
    plt.close()

if __name__ == '__main__':

    '''
    ####data debug
    path_base = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/"
    suffix = ""
    filename = path_base+"debug/data_debug"+suffix+".txt"
    data = np.array(np.loadtxt(filename, dtype=float))

    N = len(data)
    x = data[:, 0:0+Dim]
    v = data[:, Dim:Dim+Dim]
    xnorm = add.norm_l(x, axis=1)
    vnorm = add.norm_l(v, axis=1)
    
    N_subplot = Dim+2
    name_subplot = ["r ~ potenial", "r ~ forces_x", "r ~ forces_y", 
        "r ~ forces_z", "r ~ forces norm"]
    N_curve = 6
    name_curve = [
        "Pot_NFW_Exc", "Pot_NFW_ME", "Pot_DPL_ME", 
        "Pot_Sersic_ME", "Pot_Nbody_SCF", "Pot_Nbody_DS"
    ]

    idx = Dim*2
    P = list(range(N_curve))
    for i in np.arange(N_curve):
        P[i] = np.zeros((N, N_subplot))
        P[i][:,0] = data[:,idx]*1.
        # P[i][:,0] = np.abs(data[:,idx]) #or deepcopy
        P[i][:,1:4] = data[:,idx+1:idx+4]
        P[i][:,4] = add.norm_l(P[i][:,1:4], axis=1)
        idx += (N_subplot-1)
        print(P[i][1])

    # fig 0
    compare_scatter_line(P, N_subplot, N_curve, name_subplot, name_curve)
    '''




    ####data actions file
    suffix = ""
    # suffix = ".not_preprocessed"
    # suffix = ".ic__dice_ss__2_pot__scf3"
    # suffix = ".SCF"
    # suffix = "_DP_SCF" #disjunctor
    # suffix = "_DP_DS"
    # suffix = "_nono"
    # suffix = "_noT"
    # suffix = "_noCOT"
    snapshot_id = int(argv[1]) #str[] argv[0] is "python3 pyfile.py"
    filename = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/"\
        +"aa/snapshot_%d.action.method_all"%(snapshot_id)+suffix+".txt"
    data = np.array(np.loadtxt(filename, dtype=float))

    # suffix1 = ""
    # # suffix1 = "_DP_SCF"
    # filename1 = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/"\
    #     +"aa/snapshot_%d.action.method_all"%(snapshot_id)+suffix1+".txt"
    # data1 = np.array(np.loadtxt(filename1, dtype=float))
    data1 = data

    ##xvar
    N_data = len(data)
    x = data[:, 0:0+Dim]
    v = data[:, 3:3+Dim]
    X = add.norm_l(x, axis=1)
    V = add.norm_l(v, axis=1)
    xmed = np.median(X)
    vmed = np.median(V)
    xmean = np.mean(X)
    vmean = np.mean(V)

    ##potential
    P_F = data[:, 10]
    P_D = data[:, 11]

    XX = [X, x[:,0], x[:,1], x[:,2]]
    XX_name = ["norm(x)", "x", "y", "z"]
    
    # VB = [
    #     [P_F, 1, 1], 
    #     [P_D, P_F, P_D]
    # ]
    # VB_name = [
    #     ["P_F", "1", "1"], 
    #     ["P_D", "P_F", "P_D"]
    # ]
    VB = [
        [P_F], 
        [P_D]
    ]
    VB_name = [
        ["P_F"], 
        ["P_D"]
    ]
    PC = VB[1][0]/VB[0][0]
    add.DEBUG_PRINT_V(1, PC.shape)

    N_xvar = Dim+1
    N_yvar = 1
    N_curve = 1
    # fig 2
    # compare_fraction(XX, VB, XX_name, VB_name, N_xvar, N_yvar, N_curve, fracbound=-1e1)

    ##each potential
    # fig 1
    rb = 5e2
    plt.scatter(X, np.abs(P_F), label="-P_F")
    plt.scatter(X, np.abs(P_D), label="-P_D")
    plt.xlim(1./rb, rb)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("radius")
    plt.ylabel("potential")
    plt.legend()
    # plt.show()
    plt.close()



    ####action
    iast = 28
    adur = 10
    AA_TF_FP = data[:, iast+adur*0:iast+adur*0+Dim]
    AA_OD_FP = data[:, iast+adur*1:iast+adur*1+Dim] #none
    AA_GF_FP = data[:, iast+adur*2:iast+adur*2+Dim] #none
    iast += adur*5 # = 78
    AA_TF_DP = data[:, iast+adur*0:iast+adur*0+Dim]
    AA_TF_DP1 = data1[:, iast+adur*0:iast+adur*0+Dim]
    AA_OD_DP = data[:, iast+adur*1:iast+adur*1+Dim]
    AA_GF_DP = data[:, iast+adur*2:iast+adur*2+Dim] #none

    AA_TF_FP = add.merge_array_by_hstack([AA_TF_FP, np.sum(AA_TF_FP, axis=1)])
    AA_OD_FP = add.merge_array_by_hstack([AA_OD_FP, np.sum(AA_OD_FP, axis=1)])
    AA_GF_FP = add.merge_array_by_hstack([AA_GF_FP, np.sum(AA_GF_FP, axis=1)])
    AA_TF_DP = add.merge_array_by_hstack([AA_TF_DP, np.sum(AA_TF_DP, axis=1)])
    AA_TF_DP1 = add.merge_array_by_hstack([AA_TF_DP1, np.sum(AA_TF_DP1, axis=1)])
    AA_OD_DP = add.merge_array_by_hstack([AA_OD_DP, np.sum(AA_OD_DP, axis=1)])
    AA_GF_DP = add.merge_array_by_hstack([AA_GF_DP, np.sum(AA_GF_DP, axis=1)])
    
    ##something: angular moment
    m = np.ones(N_data)
    LL = tg.angularMoment(x, v, m)
    LS = add.merge_array_by_hstack([x*v, X*V])
    LL = add.merge_array_by_hstack([LL, add.norm_l(LL, axis=1)])
    # LL = add.merge_array_by_hstack([LL, np.sum(LL, axis=1)])
    LS = np.abs(LS)
    LL = np.abs(LL)
    # add.DEBUG_PRINT_V(0, LL_xyz, LL, LS)
    Lmean = np.mean(LL[-1,:])
    Lmed = np.median(LL[-1,:])
    # add.DEBUG_PRINT_V(0, xmean, xmed, vmean, vmed, Lmean, Lmed)

    # screen:
    # bd = 1e8
    # sb, cl3, cnl = add.screen_boundary(AA_TF_FP, 1./bd, bd)
    # sb, cl4, cnl = add.screen_boundary(AA_OD_FP, 1./bd, bd)
    # sb, cl5, cnl = add.screen_boundary(AA_GF_FP, 1./bd, bd)
    # sb, cl8, cnl = add.screen_boundary(AA_TF_DP, 1./bd, bd)
    # sb1, cl81, cnl1 = add.screen_boundary(AA_TF_DP1, 1./bd, bd)
    # sb, cl9, cnl = add.screen_boundary(AA_OD_DP, 1./bd, bd)
    # sb, cl10, cnl = add.screen_boundary(AA_GF_DP, 1./bd, bd)
    # add.DEBUG_PRINT_V(1, len(cl3), len(cl4), len(cl5), \
    #     len(cl8), len(cl9), len(cl10))

    # cl = np.arange(0,N_data)
    # X = X[cl]
    # V = V[cl]
    # AA_TF_FP = AA_TF_FP[cl]
    # AA_OD_FP = AA_OD_FP[cl]
    # AA_GF_FP = AA_GF_FP[cl]
    # AA_TF_DP = AA_TF_DP[cl]
    # AA_TF_DP1 = AA_TF_DP1[cl]
    # AA_OD_DP = AA_OD_DP[cl]
    # AA_GF_DP = AA_GF_DP[cl]

    # bd = 1e8-1.
    bd = 1e6
    cols = [0,1,2]
    AA_TF_FP, cl_TF_FP, cln = add.screen_boundary_some_cols(AA_TF_FP, cols, 0., bd, value_discard=bd*1e4)
    AA_TF_DP, cl_TF_DP, cln = add.screen_boundary_some_cols(AA_TF_DP, cols, 0., bd, value_discard=bd*1e4)
    AA_OD_DP, cl_OD_DP, cln = add.screen_boundary_some_cols(AA_OD_DP, cols, 0., bd, value_discard=bd*1e4)
    AA_GF_DP, cl_GF_DP, cln = add.screen_boundary_some_cols(AA_GF_DP, cols, 0., bd, value_discard=bd*1e4)
    print("bd fraction: %f, %f", len(AA_TF_FP)/N_data, len(AA_TF_DP)/N_data)

    # AA_TF_FP = AA_TF_FP[cl_TF_FP]
    # AA_TF_DP = AA_TF_DP[cl_TF_DP]
    # AA_OD_DP = AA_OD_DP[cl_OD_DP]
    # AA_GF_DP = AA_GF_DP[cl_GF_DP]

    #:: action data scale -- median
    As_TF_FP = np.median(AA_TF_FP, axis=0)
    As_OD_FP = np.median(AA_OD_FP, axis=0)
    As_GF_FP = np.median(AA_GF_FP, axis=0)
    As_TF_DP = np.median(AA_TF_DP, axis=0)
    As_TF_DP1 = np.median(AA_TF_DP1, axis=0)
    As_OD_DP = np.median(AA_OD_DP, axis=0)
    As_GF_DP = np.median(AA_GF_DP, axis=0)
    As = [[As_TF_FP, As_OD_FP, As_GF_FP], 
        [As_TF_DP, As_OD_DP, As_GF_DP]]
    # add.DEBUG_PRINT_V(1, 
    #     As_TF_FP, As_OD_FP, As_GF_FP, 
    #     As_TF_DP, As_OD_DP, As_GF_DP
    # )

    ####: plot 2d
    ##something: angluarMoment plot
    pointsize = 3.
    fontsize = 9.
    # bd = 2e5 #disjunctor
    bd = 1e6
    xplot = X #disjunctor
    # xplot = V
    i_yplot = 0 #J_lambda or L_x #disjunctor
    # i_yplot = 1
    # i_yplot = 2
    # i_yplot = -1 #J_total or L_norm
    i_yplot_name = ["J_lambda or x", "J_mu or y", "J_nu or z", "total or l2norm"]
    
    plt.scatter(xplot[cl_TF_FP], AA_TF_FP[:,i_yplot], s=pointsize, label="each "
        "action by AA_TF_FP, \t\tmedian=%e (count %d)"%(np.mean(AA_TF_FP[:,i_yplot]
        [(AA_TF_FP[:,i_yplot]>1./bd)&(AA_TF_FP[:,i_yplot]<bd)]), len(cl_TF_FP)))
    plt.scatter(xplot[cl_TF_DP], AA_TF_DP[:,i_yplot], s=pointsize, label="each "
        "action by AA_TF_DP, \t\tmedian=%e (count %d)"%(np.mean(AA_TF_DP[:,i_yplot]
        [(AA_TF_DP[:,i_yplot]>1./bd)&(AA_TF_DP[:,i_yplot]<bd)]), len(cl_TF_DP)))
    # plt.scatter(xplot, AA_TF_DP1[:,i_yplot], s=pointsize, label="each "
    #     "action by AA_TF_DP1(SCF), \t\tmedian=%e"%(np.mean(AA_TF_DP1[:,i_yplot]
    #     [(AA_TF_DP1[:,i_yplot]>1./bd)&(AA_TF_DP1[:,i_yplot]<bd)])))
    # plt.scatter(xplot, AA_OD_DP[:,i_yplot], s=pointsize, label="each "
    #     "action by AA_OD_DP, \t\tmedian=%e"%(np.mean(AA_OD_DP[:,i_yplot]
    #     [(AA_OD_DP[:,i_yplot]>1./bd)&(AA_OD_DP[:,i_yplot]<bd)])))
    # plt.scatter(xplot, LS[:,i_yplot], s=pointsize, label="each "
    #     "XV, \t\tmedian=%e"%(np.mean(LS[:,i_yplot])))
    plt.scatter(xplot, LL[:,i_yplot], s=pointsize, label="each "
        "angularMoment, \t\tmedian=%e (count %d)"
        %(np.mean(LL[:,i_yplot]), len(LL)))
    
    plt.legend()
    # plt.xlim(19., 20.)
    # plt.ylim(0., 13000.)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("radius")
    plt.ylabel("action or angularMoment (%s)"%(i_yplot_name[i_yplot]))
    plt.show()
    plt.close()
    exit(0)

    ####: plot 3d
    ##DF_x_mass and DF_AA_one
    path_DF = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/"\
        "galaxy_general/aa/snapshot_3_DF_x_mass.txt"
    DF = np.loadtxt(path_DF)
    cb = np.log10(DF[:, -1])
    rx = add.norm_l(x[:,0:3], axis=1)
    add.DEBUG_PRINT_V(1, "percentile of rx:", np.percentile(rx, [0.5, 20., 50., 80., 99.5]))
    add.DEBUG_PRINT_V(1, "percentile of DF:", np.percentile(cb, [0.5, 20., 50., 80., 99.5]))
    ddl = [
        [x, cb, "xyz"]
    ]
    PLOT = fff.Plot_model_fit()
    PLOT.plot_x_scatter3d_dd(ddl)

    path_DF = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/"\
        "galaxy_general/aa/snapshot_3_DF_AA_one.txt"
    DF = np.loadtxt(path_DF)
    cb = np.log10(DF[:, -1])
    cb = cb[cl_TF_DP]
    rJ = add.norm_l(AA_TF_DP[:,0:3], axis=1)
    add.DEBUG_PRINT_V(1, "percentile of rJ:", np.percentile(rJ, [0.5, 20., 50., 80., 99.5]))
    add.DEBUG_PRINT_V(1, "percentile of DF:", np.percentile(cb, [0.5, 20., 50., 80., 99.5]))
    ddl = [
        [AA_TF_DP, cb, "JlJmJn"]
    ]
    # bds = [1e6, 2e4, 2e4]
    bds = [5e5, 1e5, 1e5]
    PLOT = fff.Plot_model_fit()
    PLOT.plot_x_scatter3d_dd(ddl, is_lim=True, bd=bds)





    # ##to plot
    # XX = [X, V]
    # XX_name = ["norm(x)", "norm(v)"]

    # # VB = [
    # #     [AA_TF_FP, AA_OD_FP, AA_GF_FP], 
    # #     [AA_TF_DP, AA_OD_DP, AA_GF_DP]
    # # ]
    # VB = [
    #     [AA_TF_FP, AA_TF_FP, AA_TF_DP, AA_GF_FP, AA_GF_FP], 
    #     [AA_TF_DP, AA_OD_DP, AA_OD_DP, AA_TF_DP, AA_OD_DP]
    # ]
    # VB_name = [
    #     ["AA_TF_FP", "AA_TF_FP", "AA_TF_DP", "AA_GF_FP", "AA_GF_FP"], 
    #     ["AA_TF_DP", "AA_OD_DP", "AA_OD_DP", "AA_TF_DP", "AA_OD_DP"]
    # ]

    # N_xvar = 2
    # N_yvar = Dim+1
    # N_curve = 5
    # # fig 4
    # # compare_fraction(XX, VB, XX_name, VB_name, N_xvar, N_yvar, N_curve)

    # ##each action
    # VB = [
    #     [AA_TF_FP, AA_OD_FP, AA_GF_FP], 
    #     [AA_TF_DP, AA_OD_DP, AA_GF_DP]
    # ]
    # VB_name = [
    #     ["AA_TF_FP", "AA_OD_FP", "AA_GF_FP"], 
    #     ["AA_TF_DP", "AA_OD_DP", "AA_GF_DP"]
    # ]
    # As = As
    # # fig 3
    # rb = 5e2
    # ab = 5e5
    # # plt.scatter(X, np.abs(AA_TF_FP[:,0]), label="AA_TF_FP", color="red")
    # # plt.scatter(X, np.abs(AA_OD_DP[:,0]), label="AA_TEPPOD_DP", color="green")
    # # plt.scatter(X, np.abs(AA_TF_DP[:,0]), label="AA_TF_DP", color="olive")
    # for i in np.arange(2):
    #     for j in np.arange(3):
    #         plt.subplot(3,2,j*2+i+1)
    #         plt.scatter(X, VB[i][j][:,0], label=VB_name[i][j], color=colors[i])
    #         plt.xlim(1.e0, rb)
    #         plt.ylim(1.e-1, ab)
    #         plt.xscale("log")
    #         plt.yscale("log")
    #         plt.legend()
    #         plt.xlabel("radius")
    #         plt.ylabel("action lambda")
    #         print("A 0 0: ", As[i][j], VB[i][j][0,0])
    # plt.show()
    # plt.close()
