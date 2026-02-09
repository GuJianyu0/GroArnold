#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

##:: some constant variables: scale length and velocity, we set them as 1.
a_H = 1.
v_H = 1.

##:: some functions: fequencies
def Omega(r_tilde):
    return v_H/a_H *1./r_tilde*(1.-1./r_tilde*np.arctan(r_tilde))**0.5

def kappa(r_tilde):
    return v_H/a_H *1./r_tilde*(2.-1./r_tilde*np.arctan(r_tilde)-1./(1.+r_tilde**2))**0.5

def Omegakappa(r_tilde):
    return Omega(r_tilde)-0.5*kappa(r_tilde)

if __name__ == '__main__':

    # r = np.linspace(1.e-2, 5.e0, 1000)
    r = np.linspace(1.e-1, 5.e2, 10000)
    r_tidle = r/a_H #dimensionless scaled radius

    O_r = Omega(r_tidle)
    K_r = kappa(r_tidle)
    OK_r = Omegakappa(r_tidle)

    plt.plot(r_tidle,O_r, label=r"$\Omega$")
    plt.plot(r_tidle,K_r, label=r"$\kappa$")
    plt.plot(r_tidle,OK_r, label=r"$\Omega-1/2\kappa$")
    plt.plot(r_tidle,np.zeros(len(r_tidle)), color="k", label="line of zero to compare")
    plt.xlabel(r"dimensionless scaled radius $\~r=r/a_H$", fontsize=20)
    plt.ylabel(r"frequencies, setting $a_H=1\mathrm{kpc},\,v_H=1\mathrm{km/s}$", fontsize=20)
    plt.xscale("log")
    plt.legend(fontsize=20)
    plt.show()