#!/usr/bin/env python
# -*- coding:utf-8 -*-

from glob import escape
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
# import math
# import scipy
import scipy.special as spsp #gamma, gammainc

import analysis_data_distribution as add
import galaxy_models as gm
import fit_rho_fJ as fff



FPG = 4.*np.pi*gm.G

def bn_fit_(n):
    return 3.*n-1./3+0.0079/n
def bn_(n):
    # return 1. #??: by normalization
    return bn_fit_(n)

def E_estimate_scaleParameterize_(n, r):
    bn = bn_(n)
    rtrn = FPG *n*bn**(-3.*n) *r**(-1) \
        *( 1.5 *spsp.gammainc(3.*n, bn*r**(1./n))*spsp.gamma(3.*n) \
        - bn**(n)*r *spsp.gammainc(2.*n, bn*r**(1./n))*spsp.gamma(2.*n) )
    return rtrn

def L_estimate_scaleParameterize_(n, r):
    bn = bn_(n)
    rtrn = r * ( FPG *n*bn**(-3.*n) *r**(-1) \
        * spsp.gammainc(3.*n, bn*r**(1./n))*spsp.gamma(3.*n) )**0.5
    return rtrn



if __name__ == '__main__':

    ##settins
    rs = 20.
    rs_unit = 1.
    r = np.linspace(0.001*rs_unit, 100.*rs_unit, 1000) #r_ksi

    vs = 400.
    v_estimate = 1. #np.sqrt(np.abs(r*d_Phi_frac_d_r)) #??

    M_total = 137. #to refer
    rhos_simple_converge = M_total/rs**3/2.
    Itg_r = 1. #?? integrate n
    rhos = M_total/Itg_r
    rhos = rhos_simple_converge
    L_unit = np.sqrt(gm.G*M_total*rs)
    print(rs, vs, M_total, rhos, L_unit)

    ##calculate
    Phi_ = rhos*1. #gm.density_Sersic()
    Phi_estimate = 1. #??
    K_estimate = 0.5*v_estimate**2
    # virial and a strong hypothesis

    na = np.linspace(0.1, 10., 100)
    n = na[20]

    g1 = spsp.gammainc(n*2, r)*spsp.gamma(n*2)
    g2 = spsp.gammainc(n*3, r)*spsp.gamma(n*2)
    bn = bn_(n)
    # print(g1, g2, bn)
    add.DEBUG_PRINT_V(1, 4.*np.pi*n*bn**(-3.*n)* spsp.gammainc(3.*n, bn*rs**(1./n))*spsp.gamma(3.*n))

    ##fit by series
    # scale??, plot, fit, judge

    ##plot
    # plt.plot(r, g1, label="g1")
    # plt.plot(r, g2, label="g2")

    E_estimate = E_estimate_scaleParameterize_(n, r) #dimless
    J_estimate = L_estimate_scaleParameterize_(n, r)
    E_estimate /= rhos #only dimless r #v Gauss, orbit v *= 2
    J_estimate /= np.sqrt(rhos)
    plt.plot(r, -E_estimate, label="E_estimate")
    plt.plot(r, J_estimate, label="J_estimate")
    # plt.plot(r, np.abs(E_estimate)/J_estimate, label="frac")

    # E_estimate = E_estimate_scaleParameterize_(na, rs)
    # J_estimate = L_estimate_scaleParameterize_(na, rs)
    # # E_estimate /= rhos #only dimless r #v Gauss, orbit v *= 2
    # # J_estimate /= np.sqrt(rhos)
    # plt.plot(na, -E_estimate, label="E_estimate~n")
    # plt.plot(na, J_estimate, label="J_estimate~n")

    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
