#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import galaxy_models as gm
import analysis_data_distribution as add



if __name__ == '__main__':

    #### about snapshots
    N_snapshot = 10
    ss = [i*10 for i in np.arange(N_snapshot)]
    foldername = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general_DPL_less_20221107/aa/"
    filename_axisratios = foldername+"axisratios.txt"



    #### load data
    datalist = list(range(N_snapshot))
    for i_ss in np.arange(N_snapshot):
        add.DEBUG_PRINT_V(1, i_ss, "i_ss")
        filename = foldername+"snapshot_%d.action.method_all.txt"%(ss[i_ss])
        import RW_data_CMGD as rdc
        RG = rdc.Read_galaxy_data(filename)
        RG.AAAA_set_particle_variables(
            col_particle_IDs=7-1, col_particle_mass=8-1
        )

        data = RG.data
        mass = RG.particle_mass
        IDs = RG.particle_IDs
        Dim = gm.Dim #3
        iast = 28
        adur = 10
        AA_TF_FP = data[:, iast+adur*0:iast+adur*0+adur]
        AA_OD_FP = data[:, iast+adur*1:iast+adur*1+adur] #none
        AA_GF_FP = data[:, iast+adur*2:iast+adur*2+adur] #none
        iast += adur*5 #78
        AA_TF_DP = data[:, iast+adur*0:iast+adur*0+adur]
        AA_OD_DP = data[:, iast+adur*1:iast+adur*1+adur]
        AA_GF_DP = data[:, iast+adur*2:iast+adur*2+adur] #none

        AA_method = AA_TF_DP
        Act = AA_method[:, 0:3]
        Ang = AA_method[:, 3+1:7]
        Fre = AA_method[:, 7:10]
        add.DEBUG_PRINT_V(1, AA_TF_DP.shape, Act.shape, Fre.shape)
        AA = np.hstack((Act, Fre))
        cols = [0,1,2]
        # bd = 2e2
        # bd = 2e5
        bd = 1e6
        AA_cl, cl, cln = add.screen_boundary_some_cols(AA, cols, 1./bd, bd, value_discard=bd*1e4)
        # ps = AA #now: ps, mass, cl, meds
        # tgts = ps[cl] #note: used cl
        # DF = None

        x = data[:,0:3]
        v = data[:,3:6]
        potd = data[:,11]
        # aa = AA_cl
        aa = np.abs(AA_cl)
        a1 = aa[:,0]
        a2 = aa[:,1]
        a3 = aa[:,2]
        fq1 = aa[:,3]
        fq2 = aa[:,4]
        fq3 = aa[:,5]

        AA_cl[:,0:3] = aa[:,0:3] #select
        # # AA_cl[:,0:3] = aa[:,0:3]*aa[:,3:6] #select
        # AA_cl[:,0] = aa[:,0] #select
        # AA_cl[:,1] = aa[:,1]*aa[:,4]/aa[:,3]
        # AA_cl[:,2] = aa[:,2]*aa[:,5]/aa[:,3]

        # import triaxialize_galaxy as tg
        # L = tg.angularMoment(x, v)
        # lnorm = add.norm_l(L, axis=1)
        # lnorm = lnorm[cl]
        # energytransl1 = (0.5*add.norm_l(v,axis=1)**2 + potd)[cl]
        # energytransl = (energytransl1 + np.max(np.abs(potd)))
        # ac1 = gm.AA_combination_sum(aa)
        # # actioncomb = gm.AA_combination_sumWeightFrequency(aa)
        # actioncomb = gm.AA_combination_sumWeightFrequency_rateF1(aa)
        # add.DEBUG_PRINT_V(1, np.shape(energytransl), np.shape(lnorm), np.shape(actioncomb), "var shape")
        # add.DEBUG_PRINT_V(1, np.min(energytransl), np.min(actioncomb), "var min")
        # add.DEBUG_PRINT_V(1, np.median(energytransl), np.median(fq1), np.median(a3), np.median(ac1), np.median(actioncomb), "var median")

        qa = add.axis_ratio_by_configuration_data(x)
        means = add.averages_by_angle_action_data(AA_cl)
        percentiles = add.percentiles_by_angle_action_data(AA_cl)
        meds = [percentiles[i][3] for i in range(len(percentiles))]
        sx = np.sum(AA_cl[:,0]**2)
        sy = np.sum(AA_cl[:,1]**2)
        sz = np.sum(AA_cl[:,2]**2)
        means_of_square_ratio = [1., (sy/sx)**0.5, (sz/sx)**0.5]
        
        datalist[i_ss] = np.hstack((qa, means, meds, means_of_square_ratio)) #indexes from 0, 3, 12, 21, end 23
        add.DEBUG_PRINT_V(1, datalist[i_ss], "datalist[i_ss]")

    datalist = np.array(datalist)
    np.savetxt(filename_axisratios, datalist)



    #### plot data
    datalist = np.loadtxt(filename_axisratios)
    # S_start = 0
    # S_start = 2
    S_start = 4
    lt = list(range(S_start, N_snapshot))
    qy = datalist[lt,1]/datalist[lt,0]
    qz = datalist[lt,2]/datalist[lt,0]
    qzy = datalist[lt,2]/datalist[lt,1]
    mean_Jm_frac_as_Jl = datalist[lt,4]/datalist[lt,3]
    mean_Jn_frac_as_Jl = datalist[lt,5]/datalist[lt,3]
    mean_Jn_frac_as_Jm = datalist[lt,5]/datalist[lt,4]
    mean_square_Jm_frac_as_Jl = datalist[lt,22]/datalist[lt,21]
    mean_square_Jn_frac_as_Jl = datalist[lt,23]/datalist[lt,21]
    mean_square_Jn_frac_as_Jm = datalist[lt,23]/datalist[lt,22]

    qls_m = (mean_square_Jm_frac_as_Jl[-1]-mean_square_Jm_frac_as_Jl[0])/(qy[-1]-qy[0])
    qls_n = (mean_square_Jn_frac_as_Jl[-1]-mean_square_Jn_frac_as_Jl[0])/(qz[-1]-qz[0])
    qls_nm = (mean_square_Jn_frac_as_Jm[-1]-mean_square_Jn_frac_as_Jm[0])/(qzy[-1]-qzy[0])
    add.DEBUG_PRINT_V(1, qls_m, qls_n, qls_nm, "rateline")

    fontsize = 20.
    # plt.scatter(qy, mean_Jm_frac_as_Jl, label="qy ~ mean_Jm_frac_as_Jl")
    # plt.scatter(qy[0], mean_Jm_frac_as_Jl[0], color="k")
    # plt.plot(qy, mean_Jm_frac_as_Jl)
    # plt.scatter(qz, mean_Jn_frac_as_Jl, label="qz ~ mean_Jn_frac_as_Jl")
    # plt.scatter(qz[0], mean_Jn_frac_as_Jl[0], color="k")
    # plt.plot(qz, mean_Jn_frac_as_Jl)
    # plt.scatter(qzy, mean_Jn_frac_as_Jm, label="qzy ~ mean_Jn_frac_as_Jm")
    # plt.scatter(qzy[0], mean_Jn_frac_as_Jm[0], color="k")
    # plt.plot(qzy, mean_Jn_frac_as_Jm)

    plt.scatter(qy, mean_square_Jm_frac_as_Jl, label="qy ~ mean_square_Jm_frac_as_Jl")
    plt.scatter(qy[0], mean_square_Jm_frac_as_Jl[0], color="k")
    plt.plot(qy, mean_square_Jm_frac_as_Jl)
    plt.scatter(qz, mean_square_Jn_frac_as_Jl, label="qz ~ mean_square_Jn_frac_as_Jl")
    plt.scatter(qz[0], mean_square_Jn_frac_as_Jl[0], color="k")
    plt.plot(qz, mean_square_Jn_frac_as_Jl)
    plt.scatter(qzy, mean_square_Jn_frac_as_Jm, label="qzy ~ mean_square_Jn_frac_as_Jm")
    plt.scatter(qzy[0], mean_square_Jn_frac_as_Jm[0], color="k")
    plt.plot(qzy, mean_square_Jn_frac_as_Jm)

    plt.legend(fontsize=fontsize)
    plt.xlabel("axis ratio", fontsize=fontsize)
    plt.ylabel("actions ratio", fontsize=fontsize)
    # plt.xscale("log")
    # plt.yscale("log")

    plt.subplot(2,2,1)
    plt.scatter(qy, mean_square_Jm_frac_as_Jl, label="qy ~ mean_square_Jm_frac_as_Jl")
    plt.scatter(qy[0], mean_square_Jm_frac_as_Jl[0], color="k")
    plt.plot(qy, mean_square_Jm_frac_as_Jl)
    plt.legend(fontsize=fontsize)
    plt.xlabel("axis ratio", fontsize=fontsize)
    plt.ylabel("actions ratio", fontsize=fontsize)

    plt.subplot(2,2,2)
    plt.scatter(qz, mean_square_Jn_frac_as_Jl, label="qz ~ mean_square_Jn_frac_as_Jl")
    plt.scatter(qz[0], mean_square_Jn_frac_as_Jl[0], color="k")
    plt.plot(qz, mean_square_Jn_frac_as_Jl)
    plt.legend(fontsize=fontsize)
    plt.xlabel("axis ratio", fontsize=fontsize)
    plt.ylabel("actions ratio", fontsize=fontsize)

    plt.subplot(2,2,3)
    plt.scatter(qzy, mean_square_Jn_frac_as_Jm, label="qzy ~ mean_square_Jn_frac_as_Jm")
    plt.scatter(qzy[0], mean_square_Jn_frac_as_Jm[0], color="k")
    plt.plot(qzy, mean_square_Jn_frac_as_Jm)
    plt.legend(fontsize=fontsize)
    plt.xlabel("axis ratio", fontsize=fontsize)
    plt.ylabel("actions ratio", fontsize=fontsize)

    plt.show()
