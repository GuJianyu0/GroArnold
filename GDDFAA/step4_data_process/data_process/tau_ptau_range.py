#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sys import argv

import analysis_data_distribution as add
import galaxy_models as gm
import fit_rho_fJ as fff



if __name__ == '__main__':

    fig, axes = plt.subplots(2, 2)
    nameswit = [r"\lambda", r"\mu", r"\nu"]
    # swit = 0 #int(argv[1])
    # for swit in np.arange(3):
    #     plt.subplot(3, 1, swit+1)
    for swit in np.arange(1):
        plt.subplot(1, 1, swit+1)
        filename_direct_tp_orb = "~/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/action_direct_tp_orb_tau%d.debug.txt"%(swit)
        data_direct_tp_orb = np.array(np.loadtxt(filename_direct_tp_orb, dtype=float))

        data_direct_tp_orb = data_direct_tp_orb[0:-1]
        # index_percent_init  = 0.
        index_percent_init  = 0.1 #0.
        index_percent_final = index_percent_init+0.16

        alpha = -6.
        beta = -4.
        gamma = -1.
        xv = data_direct_tp_orb[:, 0:6]
        tp = data_direct_tp_orb[:, 6:12]
        tau = tp[:, 0+swit]
        ptau = tp[:, 3+swit]

        m = np.median(np.abs(ptau))*10
        err = 1e-10
        wg = np.where(np.abs(ptau)>err)[0]
        print(wg)
        print(len(wg))
        # tau = tau[wg]
        # ptau = ptau[wg]



        # filename_F = "~/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/action_range_TF_tau%d.debug.txt" %(swit)
        # filename_F = "~/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/action_range_TD_tau%d.debug.txt" %(swit)
        # filename_F = "~/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/action_range_TF_tau_whenaction.debug.txt" %(swit)
        filename_F = "~/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/action_range_TD_tau_whenaction.debug.txt" %(swit)
        data_F = np.array(np.loadtxt(filename_F, dtype=float))
        # data_D = np.array(np.loadtxt(filename_D, dtype=float))
        # data_F = data_F[ 0:16]
        # data_F = data_F[16:32]
        data_F = data_F[32:48]
        
        tau_FD = data_F[:,5]
        ptau_root_F = data_F[:,7]
        # ptau_root_D = data_D[:,7]
        # # ptau_root_D -= ptau_root_D[0]
        # print(ptau_root_D[0])
        ptau_F = data_F[:,8]
        # ptau_D = data_D[:,8]

        # err = 0.1
        # wl = np.array([0])
        # if swit==0:
        #     wl = np.where(tau_FD<-alpha+err)[0]
        # elif swit==1:
        #     wl = np.where((tau_FD>-alpha-err) & (tau_FD<-beta+err))[0]
        # else:
        #     wl = np.where(tau_FD>-gamma-err)[0]
        # tau_FD = tau_FD[wl]
        # ptau_F = ptau_F[wl]
        # ptau_F = ptau_F/abs(ptau_F)*np.log(abs(ptau_F))



        # filename_FO = "~/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/action_range_FO.debug.txt"
        # filename_orb = "~/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/action_range_orb.debug.txt"
        # data_FO = np.array(np.loadtxt(filename_FO, dtype=float))
        # data_orb = np.array(np.loadtxt(filename_orb, dtype=float))

        # alpha_FO = data_FO[0,6]
        # beta_FO = data_FO[0,7]
        # gamma_FO = data_FO[0,8]
        # tau_FO = data_FO[:,9]
        # ptau_now_FO = data_FO[:,12]
        # chi_now_FO = data_FO[:,15]
        # E_now_FO = data_FO[:,17]
        # Jt_now_FO = data_FO[:,18]
        # Kt_now_FO = data_FO[:,21]
        # ptau_root_FO = data_FO[:,-3]
        # # ptau_root_FO -= ptau_root_FO[0]
        # print(ptau_root_FO[0])
        # ptau_integ_FO = data_FO[:,-2]
        # JtKt_now_FO = -tau_FO*Jt_now_FO+Kt_now_FO
        # Ett_FO = tau_FO*tau_FO*E_now_FO
        # Pt_FO = tau_FO*tau_FO*data_FO[:,16]
        # Jtt_FO = -tau_FO*Jt_now_FO
        # I_FO = Ett_FO+Jtt_FO+JtKt_now_FO
        # B_aboutPhi_FO = -data_FO[:,9]*(data_FO[:,10]+data_FO[:,11])*data_FO[:,16]

        # alpha_orb = data_orb[0,6]
        # beta_orb = data_orb[0,7]
        # gamma_orb = data_orb[0,8]
        # tau_orb = data_orb[:,9]
        # ptau_now_orb = data_orb[:,12]
        # chi_now_orb = data_orb[:,15]
        # E_now_orb = data_orb[:,17]
        # Jt_now_orb = data_orb[:,18]
        # Kt_now_orb = data_orb[:,21]
        # ptau_root_orb = data_orb[:,-3]
        # # ptau_root_orb -= ptau_root_orb[0]
        # print(ptau_root_orb[0])
        # ptau_integ_orb = data_orb[:,-2]
        # JtKt_now_orb = -tau_orb*Jt_now_orb+Kt_now_orb
        # Ett_orb = tau_orb*tau_orb*E_now_orb
        # Pt_orb = tau_orb*tau_orb*data_orb[:,16]
        # Jtt_orb = -tau_orb*Jt_now_orb
        # I_orb = Ett_orb+Jtt_orb+JtKt_now_orb
        # B_aboutPhi_orb = -data_orb[:,9]*(data_orb[:,10]+data_orb[:,11])*data_orb[:,16]



        # pt0 = 2e0 #30
        # pt0 = 2e10 #30
        # pt0 = 2e12 #ID_305(type32, improve)
        # pt0 = 2e10 #ID_3320(type32, improve and osc valley) 9973(much less samples, type22, well) 4282(type11, improve)
        # pt0 = 2e8 #ID_0(type00, high, other method nan) 4286(type00, low, osc peak)
        # ID = 30

        err = 0.1
        t01 = [0., np.inf]
        if swit==2:
            t01 = [-gamma-err, -beta+err]
        elif swit==1:
            t01 = [-beta-err, -alpha+err]
        else:
            t01 = [-alpha-err, 1.e4]
        # t01 = [min(tau_FD), max(tau_FD)*1e-1]
        # t01 = [0., 10.]
        taulim = [5.e-1, 1.e4]
        pt01 = 1.e0
        ID = 28
        print(swit, taulim, pt01, ID)
        pointsize = 2.0
        fontsize = 16.0
        # index_percent_init  = 0.
        index_percent_init  = 0.1 #0.
        index_percent_final = index_percent_init+0.16

        # ddl = [[xv[:,0:3], "xyz"]]
        # PLOT = fff.Plot_model_fit()
        # PLOT.plot_x_scatter3d_xxx(ddl)

        if swit==2:
            plt.xlabel(r"ellip coor $\tau$ ($\mathrm{kpc^2}$)", fontsize=fontsize)
        if swit==1:
            plt.ylabel(r"ellip momentum $p_\tau$ ($km/s/kpc$), here $\mathrm{sign}(p_\tau) \log(\|p_\tau\|)$", fontsize=fontsize)
        plt.plot([-alpha,-alpha], [-pt01,pt01], label=r"$\tau=-a^2$", color="orange")
        plt.plot([-beta,-beta],   [-pt01,pt01], label=r"$\tau=-b^2$", color="orange")
        plt.plot([-gamma,-gamma], [-pt01,pt01], label=r"$\tau=-c^2$", color="orange")
        plt.plot([taulim[0], taulim[1]], [0.,0.], label=r"the root line, $y=0$", color="k")
        plt.text(-alpha, -pt01, r"$\tau=-a^2$")
        plt.text(-beta, -pt01, r"$\tau=-b^2$")
        plt.text(-gamma, -pt01, r"$\tau=-c^2$")



        # lo = len(tau_orb)
        # plt.plot(tau_orb[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     ptau_integ_orb[int(lo*index_percent_init):int(lo*index_percent_final)], label="orb", color="k")
        # # plt.plot(tau_FO[int(lo*index_percent_init):int(lo*index_percent_final)], 
        # #     ptau_integ_orb[int(lo*index_percent_init):int(lo*index_percent_final)], label="FO", linestyle='--')

        # wFD = np.where(tau_FD>562.)[0]
        # # wFD = np.where(tau_FD[wFD]<2200.)[0] #from 0
        # plt.plot(tau_FD[wFD], ptau_D[wFD], label="D", color="b")
        # plt.plot(tau_FD[wFD], ptau_F[wFD], label="F", color="r")
        # plt.plot(tau_FD, ptau_F, label="FormulaPotential tau%d"%(swit), color="r")

        # plt.scatter(tau_FD, ptau_root_F, label="FormulaPotential tau%d"%(swit), color="r")
        plt.scatter(tau_FD, ptau_F, label="FormulaPotential tau%d"%(swit), color="r", s=pointsize)
        # plt.xlim(min(t01),max(t01))
        plt.xlim(taulim[0], taulim[1])
        # yl = 1e9
        # plt.ylim(-yl,yl)

        # plt.scatter(tau, ptau, label="data", color="g", s=pointsize)
        # plt.scatter(tau, abs(ptau), label="data", color="g")
        plt.xscale("log")
        # plt.yscale("log")







        # plt.ylabel(r"ellip momentum and \|integrals\| in Fudge at $\lambda$ ()", fontsize=fontsize)
        # plt.plot(tau_orb[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     abs(E_now_orb[int(lo*index_percent_init):int(lo*index_percent_final)]), label="orb", color="purple")
        # plt.plot(tau_FO[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     abs(E_now_FO[int(lo*index_percent_init):int(lo*index_percent_final)]), label="FO", linestyle='--')

        # plt.plot(tau_orb[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     abs(Jt_now_orb[int(lo*index_percent_init):int(lo*index_percent_final)]), label="orb", color="b")
        # plt.plot(tau_FO[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     abs(Jt_now_FO[int(lo*index_percent_init):int(lo*index_percent_final)]), label="FO", linestyle='--')

        # plt.plot(tau_orb[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     abs(Kt_now_orb[int(lo*index_percent_init):int(lo*index_percent_final)]), label="orb Bt", color="r")
        # plt.plot(tau_FO[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     abs(Kt_now_FO[int(lo*index_percent_init):int(lo*index_percent_final)]), label="FO Bt", color="orange", linestyle='--')
        # plt.plot(tau_orb[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     Jtt_orb[int(lo*index_percent_init):int(lo*index_percent_final)], label="orb tAt", color="yellow")
        # plt.plot(tau_FO[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     Jtt_FO[int(lo*index_percent_init):int(lo*index_percent_final)], label="FO tAt", color="g", linestyle='--')
        # plt.plot(tau_orb[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     Ett_orb[int(lo*index_percent_init):int(lo*index_percent_final)], label="orb ttE", color="b")
        # plt.plot(tau_FO[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     Ett_FO[int(lo*index_percent_init):int(lo*index_percent_final)], label="FO ttE", color="purple", linestyle='--')
        # plt.plot(tau_orb[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     chi_now_orb[int(lo*index_percent_init):int(lo*index_percent_final)], label="orb chi", color="pink")
        # plt.plot(tau_FO[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     chi_now_FO[int(lo*index_percent_init):int(lo*index_percent_final)], label="FO chi", color="gray", linestyle='--')

        # plt.plot(tau_orb[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     Pt_orb[int(lo*index_percent_init):int(lo*index_percent_final)], label="orb", color="b")
        # plt.plot(tau_FO[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     Pt_FO[int(lo*index_percent_init):int(lo*index_percent_final)], label="FO", linestyle='--')
        # plt.plot(tau_orb[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     (Ett_orb-Pt_orb)[int(lo*index_percent_init):int(lo*index_percent_final)], label="orb ttE", color="b")
        # plt.plot(tau_FO[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     (Ett_FO-Pt_FO)[int(lo*index_percent_init):int(lo*index_percent_final)], label="FO ttE", linestyle='--')
        # plt.plot(tau_orb[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     (Jtt_orb+Kt_now_orb)[int(lo*index_percent_init):int(lo*index_percent_final)], label="orb tJt", color="r")
        # plt.plot(tau_FO[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     (Jtt_FO+Kt_now_FO)[int(lo*index_percent_init):int(lo*index_percent_final)], label="FO tJt", linestyle='--')
        # plt.plot(tau_orb[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     (Ett_orb+chi_now_orb+Jtt_orb+Kt_now_orb)[int(lo*index_percent_init):int(lo*index_percent_final)], label="orb Kt", color="b")
        # plt.plot(tau_FO[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     (Ett_FO+chi_now_FO+Jtt_FO+Kt_now_FO)[int(lo*index_percent_init):int(lo*index_percent_final)], label="FO Kt", linestyle='--')

        # plt.plot(tau_orb[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     I_orb[int(lo*index_percent_init):int(lo*index_percent_final)], label="orb", color="b")
        # plt.plot(tau_FO[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     I_FO[int(lo*index_percent_init):int(lo*index_percent_final)], label="FO", linestyle='--')
        # plt.plot(tau_orb[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     ptau_root_orb[int(lo*index_percent_init):int(lo*index_percent_final)], label="orb", color="k")
        # plt.plot(tau_FO[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     ptau_root_FO[int(lo*index_percent_init):int(lo*index_percent_final)], label="FO", linestyle='--')

        # plt.plot(tau_FO[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     (data_FO[:,16])[int(lo*index_percent_init):int(lo*index_percent_final)], label="FO pot", linestyle='--')
        # plt.plot(tau_orb[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     (data_orb[:,16])[int(lo*index_percent_init):int(lo*index_percent_final)], label="orb pot", color="k")

        # plt.plot(tau_FO[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     (B_aboutPhi_FO)[int(lo*index_percent_init):int(lo*index_percent_final)], label="FO", linestyle='--')
        # plt.plot(tau_orb[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     (B_aboutPhi_orb)[int(lo*index_percent_init):int(lo*index_percent_final)], label="orb", color="k")

        # plt.plot(tau_FO[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     ptau_now_orb[int(lo*index_percent_init):int(lo*index_percent_final)], label="FO", linestyle='--')
        # plt.plot(tau_orb[int(lo*index_percent_init):int(lo*index_percent_final)], 
        #     ptau_integ_orb[int(lo*index_percent_init):int(lo*index_percent_final)], label="orb", color="k")
        # plt.yscale("log")



    plt.title("particle_ID: %d, tau: %d" %(ID, swit), fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    fig_tmp = plt.gcf()
    plt.show()
    print("Fig ... Done.")
    plt.close("all")
