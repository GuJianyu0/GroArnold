#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

G = 43007.1 #(10^-10*M_\odot, kpc, km/s)

def norm_l(a,l=2):
    x=0
    for e in a:
        x += e**l
    return x**(1./l)

def phi_r(r):

    ## params
    rho0 = 0.000854 #e10*M_\odot kpc^-3
    r0 = 19.6 #kpc
    Ms = 4.*np.pi *rho0 *r0**3
    rr = r/r0
    print(Ms)

    ## model expression
    # ##power law:
    # alpha = 1.53
    # rho = rho0 *rr**alpha
    # phi = -G*Ms* r**(-alpha)

    ##NFW:
    rho = rho0 / ( rr**1 *(1+rr)**(3-1) )
    phi = -G*Ms *np.log(1.+rr)/r

    return phi

if __name__ == '__main__':

    xyz = np.array([-27.806963 -163.391571 -178.372787])
    #xyz = np.array([3., 4., 5.])
    #xyz = np.array([ 6., 2., 2.]) #FP_NFW_addothers=-191313.455629, DP_PL_addothers=-42357.616068 #r= 6.6332495807108, FP_NFW=-152716.3407784423 DP-85912.623763, FP_PL=-192191.780428
    #xyz = np.array([12., 2., 2.]) #FP_NFW_addothers=-160618.799320, DP_PL_addothers=-38847.188139 #r=12.3288280059379, FP_NFW=-137548.5764785167 DP-76927.744640, FP_PL=-74450.1818271
    r = norm_l(xyz)
    phi = phi_r(r)
    print(r, phi)