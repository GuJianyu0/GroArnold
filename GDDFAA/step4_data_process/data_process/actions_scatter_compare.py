#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import analysis_data_distribution as add
import galaxy_models as gm
import fit_rho_fJ as fff
import integrate_methods as im



if __name__ == '__main__':

    ##compare delE and errJ
    filename = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/snapshot_500.action.method_all_ggrbf.txt"
    data = np.array(np.loadtxt(filename, dtype=float))
    filename_ds = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/snapshot_500.action.method_all_ds.txt"
    data_ds = np.array(np.loadtxt(filename_ds, dtype=float))

    x = data[:, 0:3]
    v = data[:, 3:6]
    AA_FP = data[:, 30:33]
    AA_DP0 = data_ds[:, 63:66]
    AA_DP3 = data[:, 63:66]
    AA_DP_OD = data[:, 78:81]
    AA_DP_CLMN = data[:, [78,64,65]]
    # AA_DP_OD = AA_DP_CLMN

    N = len(data)
    X = add.norm_l(x, axis=1)
    V = add.norm_l(v, axis=1)

    T_limits = data[:,-6:]
    A0 = add.rate_abs_log(AA_DP0, AA_FP)
    A = add.rate_abs_log(AA_DP3, AA_FP)
    B = add.rate_abs_log(AA_DP_OD, AA_FP)
    B3 = add.rate_abs_log(AA_DP_CLMN, AA_FP)

    N_snapshots = 1+900
    d_snapshots = 1
    snapshots = [i for i in np.arange(N_snapshots)*d_snapshots]
    data_snapshots = np.zeros((N_snapshots,N,20))
    t = np.arange(N_snapshots)*0.01*d_snapshots
    for i in np.arange(N_snapshots):
        ss = snapshots[i]
        filename_snapshots = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general%s/snaps/txt/snapshot_%03d.txt" % ("", ss)
        data_snapshots[i] = np.array(np.loadtxt(filename_snapshots, dtype=float))
        print(r"Reading snapshot %d ... done."%(ss))

    V_snapshots = add.norm_l(data_snapshots[:,:,3:6], axis=2)
    # add.DEBUG_PRINT_V(0, data_snapshots[:,:,3:6].shape, V_snapshots.shape)
    P_snapshots = data_snapshots[:,:,13]
    E_snapshots = 0.5*V_snapshots**2 +P_snapshots
    E = E_snapshots[500] #500
    del_E = np.zeros(v.shape)
    for k in np.arange(3):
        for ID in np.arange(N):
            des = np.array([])
            L_down = (T_limits[ID,k*2]*100).astype(int)
            L_up = (T_limits[ID,k*2+1]*100+1).astype(int)
            # add.DEBUG_PRINT_V(1, k,ID, L_down,L_up, L_up-L_down)
            # np.append(des,1.)
            if not(min(snapshots)<=L_down<=max(snapshots) and min(snapshots)<=L_up<=max(snapshots) and L_down<L_up):
                des = np.append(des,0.)
                del_E[ID,k] = -1.
            else:
                for e in np.arange(L_down,L_up):
                    des = np.append(des,E_snapshots[e,ID])
                del_E[ID,k] = np.abs((np.min(des)-np.max(des))/np.mean(des))
                # del_E[ID,k] = np.abs(np.std(des)/np.mean(des))
            print(r"(%f, %f, %f) "%(ID,k,del_E[ID,k]))
    print(E,del_E)

    bd = 1e5
    AA_FP_sb, cl, cnl = add.screen_boundary(AA_FP, 1./bd, bd)
    N_dim = 3
    AA_FP = AA_FP[cl]
    AA_DP0 = AA_DP0[cl]
    AA_DP3 = AA_DP3[cl]
    AA_DP_OD = AA_DP_OD[cl]
    AA_DP_CLMN = AA_DP_CLMN[cl]
    A0 = A0[cl]
    A = A[cl]
    B = B[cl]
    B3 = B3[cl]
    X = X[cl]
    V = V[cl]
    E = E[cl]
    del_E = del_E[cl]

    AA_FP_comb = np.sum(AA_FP, axis=1)
    AA_DP0_comb = np.sum(AA_DP0, axis=1)
    AA_DP3_comb = np.sum(AA_DP3, axis=1)
    AA_DP_OD_comb = np.sum(AA_DP_OD, axis=1)
    AA_DP_CLMN_comb = np.sum(AA_DP_CLMN, axis=1)
    var = [X, V, E, del_E]
    var_name = ["R", "V", "E", "del_E"]
    N_var = len(var)



    ##for one time
    fig = plt.figure(dpi=300)
    pointsize = 0.1
    fontsize = 3.0
    for swit in np.arange(N_dim+1):
        for iv in np.arange(N_var):
            ax = fig.add_subplot(N_dim+1,N_var,swit*N_var+iv+1)
            if swit<=2:
                var[3] = del_E[:,swit]
                # ax.scatter(var[iv], AA_DP0[:,swit], s=pointsize, label="%s ~ D_DS, of coor_%d" %(var_name[iv], swit+1), color="green", marker="x", alpha=0.8)
                # ax.scatter(var[iv], AA_DP3[:,swit], s=pointsize, label="%s ~ D_RBF, of coor_%d" %(var_name[iv], swit+1), color="blue", marker="x", alpha=0.6)
                ax.scatter(var[iv], AA_FP[:,swit], s=pointsize, label="%s ~ F, of coor_%d" %(var_name[iv], swit+1), color="k", marker="x", alpha=1.)
                ax.scatter(var[iv], AA_DP_OD[:,swit], s=pointsize, label="%s ~ O, of coor_%d" %(var_name[iv], swit+1), color="red", marker="x", alpha=1.)
                ax.scatter(var[iv], AA_DP_CLMN[:,swit], s=pointsize, label="%s ~ CLMN, of coor_%d" %(var_name[iv], swit+1), color="orange", marker="x", alpha=1.)
                ax.set_yscale("log")

                # ax.scatter(var[iv], A0[:,swit], s=pointsize, label="%s ~ log(D_DS/F), of coor_%d" %(var_name[iv], swit+1), color="green", marker="x", alpha=0.8)
                # ax.scatter(var[iv], A[:,swit], s=pointsize, label="%s ~ log(D_RBF/F), of coor_%d" %(var_name[iv], swit+1), color="blue", marker="x", alpha=0.6)
                # ax.scatter(var[iv], B[:,swit], s=pointsize, label="%s ~ log(O/F), of coor_%d" %(var_name[iv], swit+1), color="red", marker="x", alpha=1.)
                # ax.scatter(var[iv], B3[:,swit], s=pointsize, label="%s ~ log(CLMN/F), of coor_%d" %(var_name[iv], swit+1), color="orange", marker="x", alpha=1.)
                # ax.set_ylim(-15,15)

            else:
                # ax.scatter(var[iv], AA_DP0_comb, s=pointsize, label="%s ~ D_DS, summation of actions" %(var_name[iv]), color="green", marker="x", alpha=0.8)
                # ax.scatter(var[iv], AA_DP3_comb, s=pointsize, label="%s ~ D_RBF, summation of actions" %(var_name[iv]), color="blue", marker="x", alpha=0.6)
                ax.scatter(var[iv], AA_FP_comb, s=pointsize, label="%s ~ F, summation of actions" %(var_name[iv]), color="k", marker="x", alpha=1.)
                ax.scatter(var[iv], AA_DP_OD_comb, s=pointsize, label="%s ~ O, summation of actions" %(var_name[iv]), color="red", marker="x", alpha=1.)
                ax.scatter(var[iv], AA_DP_CLMN_comb, s=pointsize, label="%s ~ CLMN, summation of actions" %(var_name[iv]), color="orange", marker="x", alpha=1.)
                ax.set_yscale("log")

                # ax.scatter(var[iv], add.rate_abs_log(AA_DP0_comb, AA_FP_comb), s=pointsize, label="%s ~ log(D_DS/F), of coor_%d" %(var_name[iv], swit+1), color="green", marker="x", alpha=0.8)
                # ax.scatter(var[iv], add.rate_abs_log(AA_DP3_comb, AA_FP_comb), s=pointsize, label="%s ~ log(D_RBF/F), of coor_%d" %(var_name[iv], swit+1), color="blue", marker="x", alpha=0.6)
                # ax.scatter(var[iv], add.rate_abs_log(AA_DP_OD_comb, AA_FP_comb), s=pointsize, label="%s ~ log(O/F), of coor_%d" %(var_name[iv], swit+1), color="red", marker="x", alpha=1.)
                # ax.scatter(var[iv], add.rate_abs_log(AA_DP_CLMN_comb, AA_FP_comb), s=pointsize, label="%s ~ log(CLMN/F), of coor_%d" %(var_name[iv], swit+1), color="orange", marker="x", alpha=1.)
                # ax.set_ylim(-15,15)
            
            stdline = 0.
            min_var = min(var[iv])
            max_var = max(var[iv])
            ax.plot([min_var, max_var], [stdline,stdline], color="black", lw=pointsize)
            stdline = np.log(0.5)/np.log(10.)
            ax.plot([min_var, max_var], [stdline,stdline], color="black", lw=pointsize)
            stdline = np.log(2.0)/np.log(10.)
            ax.plot([min_var, max_var], [stdline,stdline], color="black", lw=pointsize)
            plt.legend(fontsize=fontsize)
            plt.tick_params(labelsize=fontsize)

    plt.suptitle("In snapshot 500 and in the pseudo peroid of certain coordinate, \n"
                +"X: radical distance to the coordinate original point, "
                +"V: total speed, "
                +"E: Energy, "
                +"del_E: |min-max|/|mean| of energy. \n"
                # +"del_E: (standard_error/average) of energy. \n"
                +"F: actions by Stackel Fudge in formula potential; "
                +"D_DS: actions by Stackel Fudge in data potential by direct summation; \n"
                +"D_RBF: actions by Stackel Fudge in data potential by RBF interpolation to Gadget output potential; \n"
                +"O: actions by Direct Integration orbit data in data potential."
                +"CLMN: actions by combination method."
        , fontsize=fontsize)
    whspace = 0.5
    plt.subplots_adjust(hspace=whspace, wspace=whspace)
    plt.show()
    plt.close()

    fig = plt.figure(dpi=300)
    pointsize = 0.1
    fontsize = 3.0
    for swit in np.arange(N_dim+1):
        for iv in np.arange(N_var):
            ax=fig.add_subplot(N_dim+1,N_var,swit*N_var+iv+1)
            if swit<=2:
                var[3] = del_E[:,swit]
                ax.scatter(var[iv], AA_DP0[:,swit], s=pointsize, label="%s ~ D_DS, of coor_%d" %(var_name[iv], swit+1), color="green", marker="x", alpha=0.8)
                ax.scatter(var[iv], AA_DP3[:,swit], s=pointsize, label="%s ~ D_RBF, of coor_%d" %(var_name[iv], swit+1), color="blue", marker="x", alpha=0.6)
                ax.scatter(var[iv], AA_FP[:,swit], s=pointsize, label="%s ~ F, of coor_%d" %(var_name[iv], swit+1), color="k", marker="x", alpha=1.)
                # ax.scatter(var[iv], AA_DP_OD[:,swit], s=pointsize, label="%s ~ O, of coor_%d" %(var_name[iv], swit+1), color="red", marker="x", alpha=1.)
                # ax.scatter(var[iv], AA_DP_CLMN[:,swit], s=pointsize, label="%s ~ CLMN, of coor_%d" %(var_name[iv], swit+1), color="orange", marker="x", alpha=1.)
                ax.set_yscale("log")

                # ax.scatter(var[iv], A0[:,swit], s=pointsize, label="%s ~ log(D_DS/F), of coor_%d" %(var_name[iv], swit+1), color="green", marker="x", alpha=0.8)
                # ax.scatter(var[iv], A[:,swit], s=pointsize, label="%s ~ log(D_RBF/F), of coor_%d" %(var_name[iv], swit+1), color="blue", marker="x", alpha=0.6)
                # ax.scatter(var[iv], B[:,swit], s=pointsize, label="%s ~ log(O/F), of coor_%d" %(var_name[iv], swit+1), color="red", marker="x", alpha=1.)
                # ax.scatter(var[iv], B3[:,swit], s=pointsize, label="%s ~ log(CLMN/F), of coor_%d" %(var_name[iv], swit+1), color="orange", marker="x", alpha=1.)
                # ax.set_ylim(-15,15)

            else:
                ax.scatter(var[iv], AA_DP0_comb, s=pointsize, label="%s ~ D_DS, summation of actions" %(var_name[iv]), color="green", marker="x", alpha=0.8)
                ax.scatter(var[iv], AA_DP3_comb, s=pointsize, label="%s ~ D_RBF, summation of actions" %(var_name[iv]), color="blue", marker="x", alpha=0.6)
                ax.scatter(var[iv], AA_FP_comb, s=pointsize, label="%s ~ F, summation of actions" %(var_name[iv]), color="k", marker="x", alpha=1.)
                # ax.scatter(var[iv], AA_DP_OD_comb, s=pointsize, label="%s ~ O, summation of actions" %(var_name[iv]), color="red", marker="x", alpha=1.)
                # ax.scatter(var[iv], AA_DP_CLMN_comb, s=pointsize, label="%s ~ CLMN, summation of actions" %(var_name[iv]), color="orange", marker="x", alpha=1.)
                ax.set_yscale("log")

                # ax.scatter(var[iv], add.rate_abs_log(AA_DP0_comb, AA_FP_comb), s=pointsize, label="%s ~ log(D_DS/F), of coor_%d" %(var_name[iv], swit+1), color="green", marker="x", alpha=0.8)
                # ax.scatter(var[iv], add.rate_abs_log(AA_DP3_comb, AA_FP_comb), s=pointsize, label="%s ~ log(D_RBF/F), of coor_%d" %(var_name[iv], swit+1), color="blue", marker="x", alpha=0.6)
                # ax.scatter(var[iv], add.rate_abs_log(AA_DP_OD_comb, AA_FP_comb), s=pointsize, label="%s ~ log(O/F), of coor_%d" %(var_name[iv], swit+1), color="red", marker="x", alpha=1.)
                # ax.scatter(var[iv], add.rate_abs_log(AA_DP_CLMN_comb, AA_FP_comb), s=pointsize, label="%s ~ log(CLMN/F), of coor_%d" %(var_name[iv], swit+1), color="orange", marker="x", alpha=1.)
                # ax.set_ylim(-15,15)
            
            stdline = 0.
            min_var = min(var[iv])
            max_var = max(var[iv])
            ax.plot([min_var, max_var], [stdline,stdline], color="black", lw=pointsize)
            stdline = np.log(0.5)/np.log(10.)
            ax.plot([min_var, max_var], [stdline,stdline], color="black", lw=pointsize)
            stdline = np.log(2.0)/np.log(10.)
            ax.plot([min_var, max_var], [stdline,stdline], color="black", lw=pointsize)
            plt.legend(fontsize=fontsize)
            plt.tick_params(labelsize=fontsize)

    plt.suptitle("In snapshot 500 and in the pseudo peroid of certain coordinate, \n"
                +"X: radical distance to the coordinate original point, "
                +"V: total speed, "
                +"E: Energy, "
                +"del_E: (standard_error/average) of energy. \n"
                +"F: actions by Stackel Fudge in formula potential; "
                +"D_DS: actions by Stackel Fudge in data potential by direct summation; \n"
                +"D_RBF: actions by Stackel Fudge in data potential by RBF interpolation to Gadget output potential; \n"
                +"O: actions by Direct Integration orbit data in data potential."
                +"CLMN: actions by combination method."
        , fontsize=fontsize)
    whspace = 0.5
    plt.subplots_adjust(hspace=whspace, wspace=whspace)
    plt.show()
    plt.close()

    fig = plt.figure(dpi=300)
    pointsize = 0.1
    fontsize = 3.0
    for swit in np.arange(N_dim+1):
        for iv in np.arange(N_var):
            ax=fig.add_subplot(N_dim+1,N_var,swit*N_var+iv+1)
            if swit<=2:
                var[3] = del_E[:,swit]
                # # ax.scatter(var[iv], AA_DP0[:,swit], s=pointsize, label="%s ~ D_DS, of coor_%d" %(var_name[iv], swit+1), color="green", marker="x", alpha=0.8)
                # # ax.scatter(var[iv], AA_DP3[:,swit], s=pointsize, label="%s ~ D_RBF, of coor_%d" %(var_name[iv], swit+1), color="blue", marker="x", alpha=0.6)
                # ax.scatter(var[iv], AA_FP[:,swit], s=pointsize, label="%s ~ F, of coor_%d" %(var_name[iv], swit+1), color="k", marker="x", alpha=1.)
                # ax.scatter(var[iv], AA_DP_OD[:,swit], s=pointsize, label="%s ~ O, of coor_%d" %(var_name[iv], swit+1), color="red", marker="x", alpha=1.)
                # ax.scatter(var[iv], AA_DP_CLMN[:,swit], s=pointsize, label="%s ~ CLMN, of coor_%d" %(var_name[iv], swit+1), color="orange", marker="x", alpha=1.)
                # ax.set_yscale("log")

                # ax.scatter(var[iv], A0[:,swit], s=pointsize, label="%s ~ log(D_DS/F), of coor_%d" %(var_name[iv], swit+1), color="green", marker="x", alpha=0.8)
                # ax.scatter(var[iv], A[:,swit], s=pointsize, label="%s ~ log(D_RBF/F), of coor_%d" %(var_name[iv], swit+1), color="blue", marker="x", alpha=0.6)
                ax.scatter(var[iv], B[:,swit], s=pointsize, label="%s ~ log(O/F), of coor_%d" %(var_name[iv], swit+1), color="red", marker="x", alpha=1.)
                ax.scatter(var[iv], B3[:,swit], s=pointsize, label="%s ~ log(CLMN/F), of coor_%d" %(var_name[iv], swit+1), color="orange", marker="x", alpha=1.)
                ax.set_ylim(-15,15)

            else:
                # # ax.scatter(var[iv], AA_DP0_comb, s=pointsize, label="%s ~ D_DS, summation of actions" %(var_name[iv]), color="green", marker="x", alpha=0.8)
                # # ax.scatter(var[iv], AA_DP3_comb, s=pointsize, label="%s ~ D_RBF, summation of actions" %(var_name[iv]), color="blue", marker="x", alpha=0.6)
                # ax.scatter(var[iv], AA_FP_comb, s=pointsize, label="%s ~ F, summation of actions" %(var_name[iv]), color="k", marker="x", alpha=1.)
                # ax.scatter(var[iv], AA_DP_OD_comb, s=pointsize, label="%s ~ O, summation of actions" %(var_name[iv]), color="red", marker="x", alpha=1.)
                # ax.scatter(var[iv], AA_DP_CLMN_comb, s=pointsize, label="%s ~ CLMN, summation of actions" %(var_name[iv]), color="orange", marker="x", alpha=1.)
                # ax.set_yscale("log")

                # ax.scatter(var[iv], add.rate_abs_log(AA_DP0_comb, AA_FP_comb), s=pointsize, label="%s ~ log(D_DS/F), of coor_%d" %(var_name[iv], swit+1), color="green", marker="x", alpha=0.8)
                # ax.scatter(var[iv], add.rate_abs_log(AA_DP3_comb, AA_FP_comb), s=pointsize, label="%s ~ log(D_RBF/F), of coor_%d" %(var_name[iv], swit+1), color="blue", marker="x", alpha=0.6)
                ax.scatter(var[iv], add.rate_abs_log(AA_DP_OD_comb, AA_FP_comb), s=pointsize, label="%s ~ log(O/F), summation of actions" %(var_name[iv]), color="red", marker="x", alpha=1.)
                ax.scatter(var[iv], add.rate_abs_log(AA_DP_CLMN_comb, AA_FP_comb), s=pointsize, label="%s ~ log(CLMN/F), summation of actions" %(var_name[iv]), color="orange", marker="x", alpha=1.)
                ax.set_ylim(-15,15)
            
            stdline = 0.
            min_var = min(var[iv])
            max_var = max(var[iv])
            ax.plot([min_var, max_var], [stdline,stdline], color="black", lw=pointsize)
            stdline = np.log(0.5)/np.log(10.)
            ax.plot([min_var, max_var], [stdline,stdline], color="black", lw=pointsize)
            stdline = np.log(2.0)/np.log(10.)
            ax.plot([min_var, max_var], [stdline,stdline], color="black", lw=pointsize)
            plt.legend(fontsize=fontsize)
            plt.tick_params(labelsize=fontsize)

    plt.suptitle("In snapshot 500 and in the pseudo peroid of certain coordinate, \n"
                +"X: radical distance to the coordinate original point, "
                +"V: total speed, "
                +"E: Energy, "
                +"del_E: (standard_error/average) of energy. \n"
                +"F: actions by Stackel Fudge in formula potential; "
                +"D_DS: actions by Stackel Fudge in data potential by direct summation; \n"
                +"D_RBF: actions by Stackel Fudge in data potential by RBF interpolation to Gadget output potential; \n"
                +"O: actions by Direct Integration orbit data in data potential."
                +"CLMN: actions by combination method."
        , fontsize=fontsize)
    whspace = 0.5
    plt.subplots_adjust(hspace=whspace, wspace=whspace)
    plt.show()
    plt.close()

    fig = plt.figure(dpi=300)
    pointsize = 0.1
    fontsize = 3.0
    for swit in np.arange(N_dim+1):
        for iv in np.arange(N_var):
            ax=fig.add_subplot(N_dim+1,N_var,swit*N_var+iv+1)
            if swit<=2:
                var[3] = del_E[:,swit]
                # # ax.scatter(var[iv], AA_DP0[:,swit], s=pointsize, label="%s ~ D_DS, of coor_%d" %(var_name[iv], swit+1), color="green", marker="x", alpha=0.8)
                # # ax.scatter(var[iv], AA_DP3[:,swit], s=pointsize, label="%s ~ D_RBF, of coor_%d" %(var_name[iv], swit+1), color="blue", marker="x", alpha=0.6)
                # ax.scatter(var[iv], AA_FP[:,swit], s=pointsize, label="%s ~ F, of coor_%d" %(var_name[iv], swit+1), color="k", marker="x", alpha=1.)
                # ax.scatter(var[iv], AA_DP_OD[:,swit], s=pointsize, label="%s ~ O, of coor_%d" %(var_name[iv], swit+1), color="red", marker="x", alpha=1.)
                # ax.scatter(var[iv], AA_DP_CLMN[:,swit], s=pointsize, label="%s ~ CLMN, of coor_%d" %(var_name[iv], swit+1), color="orange", marker="x", alpha=1.)
                ax.set_yscale("log")

                ax.scatter(var[iv], A0[:,swit], s=pointsize, label="%s ~ log(D_DS/F), of coor_%d" %(var_name[iv], swit+1), color="green", marker="x", alpha=0.8)
                ax.scatter(var[iv], A[:,swit], s=pointsize, label="%s ~ log(D_RBF/F), of coor_%d" %(var_name[iv], swit+1), color="blue", marker="x", alpha=0.6)
                # ax.scatter(var[iv], B[:,swit], s=pointsize, label="%s ~ log(O/F), of coor_%d" %(var_name[iv], swit+1), color="red", marker="x", alpha=1.)
                # ax.scatter(var[iv], B3[:,swit], s=pointsize, label="%s ~ log(CLMN/F), of coor_%d" %(var_name[iv], swit+1), color="orange", marker="x", alpha=1.)
                # ax.set_ylim(-15,15)

            else:
                # # ax.scatter(var[iv], AA_DP0_comb, s=pointsize, label="%s ~ D_DS, summation of actions" %(var_name[iv]), color="green", marker="x", alpha=0.8)
                # # ax.scatter(var[iv], AA_DP3_comb, s=pointsize, label="%s ~ D_RBF, summation of actions" %(var_name[iv]), color="blue", marker="x", alpha=0.6)
                # ax.scatter(var[iv], AA_FP_comb, s=pointsize, label="%s ~ F, summation of actions" %(var_name[iv]), color="k", marker="x", alpha=1.)
                # ax.scatter(var[iv], AA_DP_OD_comb, s=pointsize, label="%s ~ O, summation of actions" %(var_name[iv]), color="red", marker="x", alpha=1.)
                # ax.scatter(var[iv], AA_DP_CLMN_comb, s=pointsize, label="%s ~ CLMN, summation of actions" %(var_name[iv]), color="orange", marker="x", alpha=1.)
                ax.set_yscale("log")

                ax.scatter(var[iv], add.rate_abs_log(AA_DP0_comb, AA_FP_comb), s=pointsize, label="%s ~ log(D_DS/F), summation of actions" %(var_name[iv]), color="green", marker="x", alpha=0.8)
                ax.scatter(var[iv], add.rate_abs_log(AA_DP3_comb, AA_FP_comb), s=pointsize, label="%s ~ log(D_RBF/F), summation of actions" %(var_name[iv]), color="blue", marker="x", alpha=0.6)
                # ax.scatter(var[iv], add.rate_abs_log(AA_DP_OD_comb, AA_FP_comb), s=pointsize, label="%s ~ log(O/F), of coor_%d" %(var_name[iv], swit+1), color="red", marker="x", alpha=1.)
                # ax.scatter(var[iv], add.rate_abs_log(AA_DP_CLMN_comb, AA_FP_comb), s=pointsize, label="%s ~ log(CLMN/F), of coor_%d" %(var_name[iv], swit+1), color="orange", marker="x", alpha=1.)
                # ax.set_ylim(-15,15)
            
            stdline = 0.
            min_var = min(var[iv])
            max_var = max(var[iv])
            ax.plot([min_var, max_var], [stdline,stdline], color="black", lw=pointsize)
            stdline = np.log(0.5)/np.log(10.)
            ax.plot([min_var, max_var], [stdline,stdline], color="black", lw=pointsize)
            stdline = np.log(2.0)/np.log(10.)
            ax.plot([min_var, max_var], [stdline,stdline], color="black", lw=pointsize)
            plt.legend(fontsize=fontsize)
            plt.tick_params(labelsize=fontsize)

    plt.suptitle("In snapshot 500 and in the pseudo peroid of certain coordinate, \n"
                +"X: radical distance to the coordinate original point, "
                +"V: total speed, "
                +"E: Energy, "
                +"del_E: (standard_error/average) of energy. \n"
                +"F: actions by Stackel Fudge in formula potential; "
                +"D_DS: actions by Stackel Fudge in data potential by direct summation; \n"
                +"D_RBF: actions by Stackel Fudge in data potential by RBF interpolation to Gadget output potential; \n"
                +"O: actions by Direct Integration orbit data in data potential."
                +"CLMN: actions by combination method."
        , fontsize=fontsize)
    whspace = 0.5
    plt.subplots_adjust(hspace=whspace, wspace=whspace)
    plt.show()
    plt.close()



    # ##plot2d all
    # fig = plt.figure(dpi=300)
    # pointsize = 0.1
    # fontsize = 3.0
    # for swit in np.arange(N_dim+1):
    #     for iv in np.arange(N_var):
    #         ax=fig.add_subplot(N_dim+1,N_var,swit*N_var+iv+1)
    #         if swit<=2:
    #             var[3] = del_E[:,swit]
    #             # ax.scatter(var[iv], AA_DP0[:,swit], s=pointsize, label="%s ~ D_DS, of coor_%d" %(var_name[iv], swit+1), color="green", marker="x", alpha=0.8)
    #             # ax.scatter(var[iv], AA_DP3[:,swit], s=pointsize, label="%s ~ D_RBF, of coor_%d" %(var_name[iv], swit+1), color="blue", marker="x", alpha=0.6)
    #             ax.scatter(var[iv], AA_FP[:,swit], s=pointsize, label="%s ~ F, of coor_%d" %(var_name[iv], swit+1), color="k", marker="x", alpha=1.)
    #             ax.scatter(var[iv], AA_DP_OD[:,swit], s=pointsize, label="%s ~ O, of coor_%d" %(var_name[iv], swit+1), color="red", marker="x", alpha=1.)
    #             ax.scatter(var[iv], AA_DP_CLMN[:,swit], s=pointsize, label="%s ~ CLMN, of coor_%d" %(var_name[iv], swit+1), color="orange", marker="x", alpha=1.)
    #             ax.set_yscale("log")

    #             # ax.scatter(var[iv], A0[:,swit], s=pointsize, label="%s ~ log(D_DS/F), of coor_%d" %(var_name[iv], swit+1), color="green", marker="x", alpha=0.8)
    #             # ax.scatter(var[iv], A[:,swit], s=pointsize, label="%s ~ log(D_RBF/F), of coor_%d" %(var_name[iv], swit+1), color="blue", marker="x", alpha=0.6)
    #             # ax.scatter(var[iv], B[:,swit], s=pointsize, label="%s ~ log(O/F), of coor_%d" %(var_name[iv], swit+1), color="red", marker="x", alpha=1.)
    #             # ax.scatter(var[iv], B3[:,swit], s=pointsize, label="%s ~ log(CLMN/F), of coor_%d" %(var_name[iv], swit+1), color="orange", marker="x", alpha=1.)
    #             # ax.set_ylim(-15,15)

    #         else:
    #             # ax.scatter(var[iv], AA_DP0_comb, s=pointsize, label="%s ~ D_DS, summation of actions" %(var_name[iv]), color="green", marker="x", alpha=0.8)
    #             # ax.scatter(var[iv], AA_DP3_comb, s=pointsize, label="%s ~ D_RBF, summation of actions" %(var_name[iv]), color="blue", marker="x", alpha=0.6)
    #             ax.scatter(var[iv], AA_FP_comb, s=pointsize, label="%s ~ F, summation of actions" %(var_name[iv]), color="k", marker="x", alpha=1.)
    #             ax.scatter(var[iv], AA_DP_OD_comb, s=pointsize, label="%s ~ O, summation of actions" %(var_name[iv]), color="red", marker="x", alpha=1.)
    #             ax.scatter(var[iv], AA_DP_CLMN_comb, s=pointsize, label="%s ~ CLMN, summation of actions" %(var_name[iv]), color="orange", marker="x", alpha=1.)
    #             ax.set_yscale("log")

    #             # ax.scatter(var[iv], add.rate_abs_log(AA_DP0_comb, AA_FP_comb), s=pointsize, label="%s ~ log(D_DS/F), of coor_%d" %(var_name[iv], swit+1), color="green", marker="x", alpha=0.8)
    #             # ax.scatter(var[iv], add.rate_abs_log(AA_DP3_comb, AA_FP_comb), s=pointsize, label="%s ~ log(D_RBF/F), of coor_%d" %(var_name[iv], swit+1), color="blue", marker="x", alpha=0.6)
    #             # ax.scatter(var[iv], add.rate_abs_log(AA_DP_OD_comb, AA_FP_comb), s=pointsize, label="%s ~ log(O/F), of coor_%d" %(var_name[iv], swit+1), color="red", marker="x", alpha=1.)
    #             # ax.scatter(var[iv], add.rate_abs_log(AA_DP_CLMN_comb, AA_FP_comb), s=pointsize, label="%s ~ log(CLMN/F), of coor_%d" %(var_name[iv], swit+1), color="orange", marker="x", alpha=1.)
    #             # ax.set_ylim(-15,15)
            
    #         stdline = 0.
    #         min_var = min(var[iv])
    #         max_var = max(var[iv])
    #         ax.plot([min_var, max_var], [stdline,stdline], color="black", lw=pointsize)
    #         stdline = np.log(0.5)/np.log(10.)
    #         ax.plot([min_var, max_var], [stdline,stdline], color="black", lw=pointsize)
    #         stdline = np.log(2.0)/np.log(10.)
    #         ax.plot([min_var, max_var], [stdline,stdline], color="black", lw=pointsize)
    #         plt.legend(fontsize=fontsize)
    #         plt.tick_params(labelsize=fontsize)

    # plt.suptitle("In snapshot 500 and in the pseudo peroid of certain coordinate, \n"
    #             +"X: radical distance to the coordinate original point, "
    #             +"V: total speed, "
    #             +"E: Energy, "
    #             +"del_E: (standard_error/average) of energy. \n"
    #             +"F: actions by Stackel Fudge in formula potential; "
    #             +"D_DS: actions by Stackel Fudge in data potential by direct summation; \n"
    #             +"D_RBF: actions by Stackel Fudge in data potential by RBF interpolation to Gadget output potential; \n"
    #             +"O: actions by Direct Integration orbit data in data potential."
    #             +"CLMN: actions by combination method."
    #     , fontsize=fontsize)
    # whspace = 0.5
    # plt.subplots_adjust(hspace=whspace, wspace=whspace)
    # plt.show()
    # plt.close()



    # ##plot2d
    # fig = plt.figure(dpi=300)
    # pointsize = 0.1
    # fontsize = 3.0
    # dpi = 500
    # for swit in np.arange(3):
    #     ax=fig.add_subplot(3,2,(swit+1)*2-1)
    #     ax.scatter(X, A[:,swit], s=pointsize, label="X_norm ~ log(D/F), of coor_%d" %(swit+1), color="blue", marker="+", alpha=0.6)
    #     ax.scatter(X, B[:,swit], s=pointsize, label="X_norm ~ log(O/F), of coor_%d" %(swit+1), color="red", marker="x", alpha=1.)
    #     stdline = 0.
    #     ax.plot([min(X), max(X)], [stdline,stdline], color="black", lw=pointsize)
    #     stdline = np.log(0.5)/np.log(10.)
    #     ax.plot([min(X), max(X)], [stdline,stdline], color="black", lw=pointsize)
    #     stdline = np.log(2.0)/np.log(10.)
    #     ax.plot([min(X), max(X)], [stdline,stdline], color="black", lw=pointsize)
    #     ax.set_ylim(-15,15)
    #     plt.legend(fontsize=fontsize)

    #     ax=fig.add_subplot(3,2,(swit+1)*2)
    #     ax.scatter(V, A[:,swit], s=pointsize, label="V_norm ~ log(D/F), of coor_%d" %(swit+1), color="blue", marker="+", alpha=0.6)
    #     ax.scatter(V, B[:,swit], s=pointsize, label="V_norm ~ log(O/F), of coor_%d" %(swit+1), color="red", marker="x", alpha=1.)
    #     stdline = 0.
    #     ax.plot([min(V), max(V)], [stdline,stdline], color="black", lw=pointsize)
    #     stdline = np.log(0.5)/np.log(10.)
    #     ax.plot([min(V), max(V)], [stdline,stdline], color="black", lw=pointsize)
    #     stdline = np.log(2.0)/np.log(10.)
    #     ax.plot([min(V), max(V)], [stdline,stdline], color="black", lw=pointsize)
    #     ax.set_ylim(-15,15)
    #     plt.legend(fontsize=fontsize)

    #     if swit==0:
    #         ax.set_title("F: Stackel Fudge in formula potential\n"
    #             +"D:Stackel Fudge in data potential\n"
    #             +"O:Direct Integration orbit data in data potential", fontsize=fontsize)
    # whspace = 0.4
    # plt.subplots_adjust(hspace=whspace, wspace=whspace)
    # plt.show()
    # plt.close()



    # ##plot3d
    # fig = plt.figure(dpi=300)
    # pointsize = 0.2
    # fontsize = 6.0
    # dpi = 500
    # ax=fig.add_subplot(1,1,1, projection='3d')

    # ax.scatter(A[:,0], A[:,1], A[:,2], s=pointsize, label="D/F")
    # ax.scatter(B[:,0], B[:,1], B[:,2], s=pointsize, label="O/F")
    # ax.plot([min(A[:,0]), max(A[:,0])], [stdline,stdline], [stdline,stdline], color="black", lw=pointsize)
    # ax.plot([stdline,stdline], [min(A[:,1]), max(A[:,1])], [stdline,stdline], color="black", lw=pointsize)
    # ax.plot([stdline,stdline], [stdline,stdline], [min(A[:,2]), max(A[:,2])], color="black", lw=pointsize)

    # # ax.set_xlim(-lim[0]*0., lim[0])
    # # ax.set_ylim(-lim[1]*0., lim[1])
    # # ax.set_zlim(-lim[2]*0., lim[2])
    # # ax.set_xscale("log")
    # # ax.set_yscale("log")
    # # ax.set_zscale("log")

    # ax.grid(True)
    # ax.set_xlabel(r"x", fontsize=fontsize)
    # ax.set_ylabel(r"y", fontsize=fontsize)
    # ax.set_zlabel(r"z", fontsize=fontsize)
    # ax.set_title("F: Stackel Fudge in formula potential\n"
    #     +"D:Stackel Fudge in data potential\n"
    #     +"O:Direct Integration orbit data in data potential", fontsize=fontsize/2)
    # # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)
    # plt.legend(fontsize=fontsize)

    # plt.show()
    # plt.close("all")


    # ## orbit 3d
    # fig = plt.figure(dpi=300)
    # pointsize = 0.1
    # fontsize = 6.0
    # dpi = 500
    # ax=fig.add_subplot(1,1,1, projection='3d')
    # for i in [28,32]:
    #     ID = i
    #     filename = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/orbit_%d_directintegration_endless.txt" % (ID)
    #     data = np.array(np.loadtxt(filename, dtype=float))
    #     t = np.arange(len(data))*0.01
    #     print(data)
    #     x0 = data[:, 0]
    #     x1 = data[:, 1]
    #     x2 = data[:, 2]
    #     r = add.norm_l(data[:,0:3], axis=1)
    #     E = data[:, -1]
    #     ax.scatter(x0,x1,x2, label=r"ID_%d"%(ID), s=pointsize)
    #     ax.plot(x0,x1,x2, lw=pointsize/2)
    #     plt.legend(fontsize=fontsize)
    # ax.scatter([0.],[0.],[0.], s=pointsize*2, color="black")
    # ax.set_xlabel(r"x", fontsize=fontsize)
    # ax.set_ylabel(r"y", fontsize=fontsize)
    # ax.set_zlabel(r"z", fontsize=fontsize)
    # whspace = 0.4
    # plt.subplots_adjust(hspace=whspace, wspace=whspace)
    # plt.show()
    # plt.close()
