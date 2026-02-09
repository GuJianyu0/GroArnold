#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import analysis_data_distribution as add
import galaxy_models as gm
import sympy as sp

def Halmitonion_Oscilator(x, v, GM, rs):
    return 1

def Halmitonion_Isochrone(x, v, GM, rs):
    return 1

def Halmitonion_PL(x, v, GM, rs):
    return 1

def Halmitonion_model_by_formula(x, v, func):
    return func(x,v)

def Halmitonion_model_by_data(data):
    return data

def Halmitonion_actions_Isochrone(J, GM, rs):
    return None

def Halmitonion_actionScale_estimate_PL(J, GM, rs, J_scale, p, b):
    Jn = J/J_scale
    return -Jn**p*np.exp(b)

def derive_1d_Euler1(x, dx, fx, *args):
    return (fx(x+dx, *args)-fx(x, *args))/dx

def DF_energy_PL(E, GM, rs):
    return 1

def DF_energy_Oscilator(E, GM, rs):
    return 1

def DF_energy_Isochrone(E, GM, rs): #?? wrong unit dimonsion
    En = -E*rs/GM
    C = 1./(2.**0.5*(2.*np.pi)**3*(GM*rs)*1.5)
    f1 = np.sqrt(En)/(2*(1.-En))**4
    f2 = 27. - 66*En + 320*En**2 - 240*En**3 + 64*En**4
    f3 = 3*(16*E**2 + 28*En -9.)
    return C*f1*(f2+f3)

def DF_energy_estimate(E, GM, rs, DF_E_ref):
    dx = 1e-6
    args = GM, rs
    DF_E_0 = DF_E_ref(E, *args)
    DF_E_prime = derive_1d_Euler1(E, dx, DF_E_ref, *args)
    delta_DF = E-DF_E_0
    return DF_E_0 + DF_E_prime*delta_DF

def DF_actionComb_estimate(JC, GM, rs, DF_E_ref):
    return None



if __name__ == "__main__":

    # #data read
    # filename = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/aa/snapshot_90.action.method_all.txt"
    # import RW_data_CMGD as rdc
    # RG = rdc.Read_galaxy_data(filename)
    # RG.AAAA_set_particle_variables(
    #     col_particle_IDs=7-1, col_particle_mass=8-1
    # )

    # data = RG.data
    # mass = RG.particle_mass
    # IDs = RG.particle_IDs
    # Dim = gm.Dim #3
    # iast = 28
    # adur = 10
    # AA_TF_FP = data[:, iast+adur*0:iast+adur*0+adur]
    # AA_OD_FP = data[:, iast+adur*1:iast+adur*1+adur] #none
    # AA_GF_FP = data[:, iast+adur*2:iast+adur*2+adur] #none
    # iast += adur*5 # = 78
    # AA_TF_DP = data[:, iast+adur*0:iast+adur*0+adur]
    # AA_OD_DP = data[:, iast+adur*1:iast+adur*1+adur]
    # AA_GF_DP = data[:, iast+adur*2:iast+adur*2+adur] #none

    # AA_method = AA_TF_DP
    # Act = AA_method[:, 0:3]
    # Ang = AA_method[:, 3+1:7]
    # Fre = AA_method[:, 7:10]
    # add.DEBUG_PRINT_V(1, AA_TF_DP.shape, Act.shape, Fre.shape)
    # AA = np.hstack((Act, Fre))
    # cols = [0,1,2]
    # # bd = 2e2
    # bd = 2e5
    # AA_cl, cl, cln = add.screen_boundary_some_cols(AA, cols, 1./bd, bd, value_discard=bd*1e4)
    # ps = AA #now: ps, mass, cl, meds
    # tgts = ps[cl] #note: used cl
    # DF = None

    # x = data[:,0:3]
    # v = data[:,3:6]
    # import triaxialize_galaxy as tg
    # L = tg.angularMoment(x, v)
    # lnorm = add.norm_l(L, axis=1)
    # potd = data[:,11]
    # # aa = AA_cl
    # aa = np.abs(AA_cl)
    # a1 = aa[:,0]
    # a2 = aa[:,1]
    # a3 = aa[:,2]
    # fq1 = aa[:,3]
    # fq2 = aa[:,4]
    # fq3 = aa[:,5]

    # energytransl1 = (0.5*add.norm_l(v,axis=1)**2 + potd)[cl]
    # energytransl = (energytransl1 + np.max(np.abs(potd)))
    # lnorm = lnorm[cl]
    # ac1 = gm.AA_combination_sum(aa)
    # # actioncomb = gm.AA_combination_sumWeightFrequency(aa)
    # actioncomb = gm.AA_combination_sumWeightFrequency_rateF1(aa)
    # add.DEBUG_PRINT_V(1, np.shape(energytransl), np.shape(lnorm), np.shape(actioncomb), "var shape")
    # add.DEBUG_PRINT_V(1, np.min(energytransl), np.min(actioncomb), "var min")
    # add.DEBUG_PRINT_V(1, np.median(energytransl), np.median(fq1), np.median(a3), np.median(ac1), np.median(actioncomb), "var median")



    # #data DF
    # H_0 = Halmitonion_Isochrone(x, v, rs)
    # H_data = Halmitonion_model_by_data(energytransl1)
    # DFE = None
    # import KDTree_python as kdtp
    # KD = kdtp.KDTree_galaxy_particles(ps[:,cols], weight_extern_instinct=mass)
    # add.DEBUG_PRINT_V(1, np.shape(tgts), "tgts")
    # DFA = KD.density_SPH(tgts[:,cols]) #some are None #?? debug

    #data prepare
    # M = np.sum(mass)
    M = 137.
    # rs = np.median(add.norm_l(x))
    rs = 100.
    # # J_scale = actionScale_by_otherScale(M, rs)
    J_scale = 1.

    #propose DF
    N_samples = 100
    J = np.logspace(np.log10(1e0), np.log10(2e5), N_samples)
    GM = gm.G*M
    p, b = 0.30019442, -5.43080785
    H_0 = None
    HJ_set_for_usual = Halmitonion_actionScale_estimate_PL(J, GM, rs, J_scale, p, b)
    # HJ_set_for_usual = -np.logspace(np.log10(1e-5), np.log10(1e2), N_samples)
    DF_E_ref = DF_energy_Isochrone
    DF_0 = DF_E_ref(HJ_set_for_usual, GM, rs)
    DFE = DF_energy_estimate(HJ_set_for_usual, GM, rs, DF_E_ref)
    DFA = DFE

    #plot
    import matplotlib.pyplot as plt
    add.DEBUG_PRINT_V(1, J, HJ_set_for_usual, DF_0, DFA, "fJ")
    # add.DEBUG_PRINT_V(1, np.median(J), np.median(HJ_set_for_usual))
    plt.subplot(3,1,1)
    plt.plot(np.log(J), np.log(-HJ_set_for_usual), label="ln(J_comb) ~ ln(-E) (unit unit)")
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(np.log(-HJ_set_for_usual), np.log(DF_0), label="ln(-E) ~ ln(DF_isochorne) (unit unit)")
    plt.plot(np.log(-HJ_set_for_usual), np.log(DFA), label="ln(-E) ~ ln(DF) (unit unit)")
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(np.log(J), np.log(DF_0), label="ln(J_comb) ~ ln(DF_isochorne) (unit unit)")
    plt.plot(np.log(J), np.log(DFA), label="ln(J_comb) ~ ln(DF) (unit unit)")
    plt.legend()
    plt.show()
