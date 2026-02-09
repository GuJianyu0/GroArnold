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

    swit = 0
    filename_direct_tp_orb = "~/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/action_direct_tp_orb_tau%d.debug.txt"%(swit)
    data_direct_tp_orb = np.array(np.loadtxt(filename_direct_tp_orb, dtype=float))

    data_direct_tp_orb = data_direct_tp_orb #[0:50]
    # index_percent_init  = 0.
    index_percent_init  = 0.1 #0.
    index_percent_final = index_percent_init+0.16

    alpha = -6.
    beta = -4.
    gamma = -1.
    xv = data_direct_tp_orb[:, 0:6]
    tp = data_direct_tp_orb[:, 6:12]
    tau = tp[:, 0]
    ptau = tp[:, 3]
    # lo = len(tau)
    # print("len(tau): ", lo)
    # m = np.median(np.abs(ptau))*10
    # wg = np.where(np.abs(ptau)>m)[0]
    # add.DEBUG_PRINT_V(1, wg, len(wg), m)
    # add.DEBUG_PRINT_V(1, ptau)

    ddl = [[xv[:,0:3], "xyz"]]
    PLOT = fff.Plot_model_fit()
    PLOT.plot_x_scatter3d_xxx(ddl)



    filename_F = "~/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/action_range_TF.debug.txt"
    filename_D = "~/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/action_range_TD.debug.txt"
    data_F = np.array(np.loadtxt(filename_F, dtype=float))
    data_D = np.array(np.loadtxt(filename_D, dtype=float))
    
    tau_FD = data_F[:,5]
    ptau_root_F = data_F[:,7]
    ptau_root_D = data_D[:,7]
    # ptau_root_D -= ptau_root_D[0]
    print(ptau_root_D[0])
    ptau_F = data_F[:,8]
    ptau_D = data_D[:,8]



    filename_FO = "~/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/action_range_FO.debug.txt"
    filename_orb = "~/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/action_range_orb.debug.txt"
    data_FO = np.array(np.loadtxt(filename_FO, dtype=float))
    data_orb = np.array(np.loadtxt(filename_orb, dtype=float))

    alpha_FO = data_FO[0,6]
    beta_FO = data_FO[0,7]
    gamma_FO = data_FO[0,8]
    tau_FO = data_FO[:,9]
    ptau_now_FO = data_FO[:,12]
    chi_now_FO = data_FO[:,15]
    E_now_FO = data_FO[:,17]
    Jt_now_FO = data_FO[:,18]
    Kt_now_FO = data_FO[:,21]
    ptau_root_FO = data_FO[:,-3]
    # ptau_root_FO -= ptau_root_FO[0]
    print(ptau_root_FO[0])
    ptau_integ_FO = data_FO[:,-2]
    JtKt_now_FO = -tau_FO*Jt_now_FO+Kt_now_FO
    Ett_FO = tau_FO*tau_FO*E_now_FO
    Pt_FO = tau_FO*tau_FO*data_FO[:,16]
    Jtt_FO = -tau_FO*Jt_now_FO
    I_FO = Ett_FO+Jtt_FO+JtKt_now_FO
    B_aboutPhi_FO = -data_FO[:,9]*(data_FO[:,10]+data_FO[:,11])*data_FO[:,16]

    alpha_orb = data_orb[0,6]
    beta_orb = data_orb[0,7]
    gamma_orb = data_orb[0,8]
    tau_orb = data_orb[:,9]
    ptau_now_orb = data_orb[:,12]
    chi_now_orb = data_orb[:,15]
    E_now_orb = data_orb[:,17]
    Jt_now_orb = data_orb[:,18]
    Kt_now_orb = data_orb[:,21]
    ptau_root_orb = data_orb[:,-3]
    # ptau_root_orb -= ptau_root_orb[0]
    print(ptau_root_orb[0])
    ptau_integ_orb = data_orb[:,-2]
    JtKt_now_orb = -tau_orb*Jt_now_orb+Kt_now_orb
    Ett_orb = tau_orb*tau_orb*E_now_orb
    Pt_orb = tau_orb*tau_orb*data_orb[:,16]
    Jtt_orb = -tau_orb*Jt_now_orb
    I_orb = Ett_orb+Jtt_orb+JtKt_now_orb
    B_aboutPhi_orb = -data_orb[:,9]*(data_orb[:,10]+data_orb[:,11])*data_orb[:,16]

    pt01 = 1.e0
    ID = 30
    # pt0 = 2e0 #30
    # pt0 = 2e10 #30
    # pt0 = 2e12 #ID_305(type32, improve)
    # pt0 = 2e10 #ID_3320(type32, improve and osc valley) 9973(much less samples, type22, well) 4282(type11, improve)
    # pt0 = 2e8 #ID_0(type00, high, other method nan) 4286(type00, low, osc peak)
    # ID = 30
    pointsize = 0.2
    fontsize = 20.0
    lo = len(tau_orb)
    # index_percent_init  = 0.
    index_percent_init  = 0.1 #0.
    index_percent_final = index_percent_init+0.16

    plt.xlabel(r"ellip coor $\lambda$ ($\mathrm{kpc^2}$)", fontsize=fontsize)
    plt.ylabel(r"ellip momentum root solve function in Fudge at $\lambda$ ($kpc^2$)", fontsize=fontsize)
    plt.title("particle_ID: %d" % ID, fontsize=fontsize)
    plt.plot([alpha,alpha], [-pt01,pt01], label=r"$\tau=-a^2$", color="orange")
    plt.plot([beta,beta],   [-pt01,pt01], label=r"$\tau=-b^2$", color="orange")
    plt.plot([gamma,gamma], [-pt01,pt01], label=r"$\tau=-c^2$", color="orange")
    plt.plot([min(tau),max(tau)], [0.,0.], label=r"the root line, $y=0$", color="k")



    wFD = np.where(tau_FD>562.)[0]
    # wFD = np.where(tau_FD[wFD]<2200.)[0] #from 0
    plt.plot(tau_FD[wFD], ptau_D[wFD], label="D", color="b")
    plt.plot(tau_FD[wFD], ptau_F[wFD], label="F", color="r")

    # plt.plot(tau, abs(ptau), label="data", color="g")
    plt.scatter(tau, ptau, label="data", color="g")
    # plt.scatter(tau, abs(ptau), label="data", color="g")
    
    plt.plot(tau_orb[int(lo*index_percent_init):int(lo*index_percent_final)], 
        ptau_integ_orb[int(lo*index_percent_init):int(lo*index_percent_final)], label="orb", color="k")
    # plt.plot(tau_FO[int(lo*index_percent_init):int(lo*index_percent_final)], 
    #     ptau_integ_orb[int(lo*index_percent_init):int(lo*index_percent_final)], label="FO", linestyle='--')



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



    plt.legend(fontsize=fontsize)
    # plt.xscale("log")
    fig_tmp = plt.gcf()
    plt.show()
    print("Fig ... Done.")
    plt.close("all")
