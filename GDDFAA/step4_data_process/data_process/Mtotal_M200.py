#!/usr/bin/env python
# -*- coding:utf-8 -*-

## to convert M_total of a galaxy to M200 for DICE IC

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root,fsolve
from scipy import integrate
from scipy.integrate import tplquad,dblquad,quad

## all units are 1e10*M_solar, kpc, km/s
G = 43007.1
h7 = 0.71
H0 = 100.*h7*1.e-3
rho_c = 3*H0**2/(8*np.pi*G) #=139.912956487e-10

## set rho_0, r_s, flat_xyz(=1.) of a potential model
rho_0 = 1. #1.806463 #e4 #8.54e6 *1e-10 #by PM17 NFW rho_0 ##1e10*M_solar * kpc^(-3)
r_s = 2.

##dice settings
dice_M200 = 130.
# rho_0 = 3*dice_M200/(4*np.pi*r_s**3)

## functions
def norm_l(a,l=2): #??
    x=0
    for e in a:
        x += e**l
    return x**(1./l)

def isSmaller(a,b):
    if a<=b:
        return 1
    else:
        return 0

def foo(x):
    return x

def rho_Plummer_(r):
    return rho_0 *(1+(r/r_s)**2)**(-2.5)

def rho_Plummer_solve_(r):
    ## r/r_s here is m=norm_2(x/hx,y/hy,z/hz) in DICE
    return rho_0 *(1+(r/r_s)**2)**(-2.5) -200.*rho_c

def r200_(func):
    # result_root = root(func,[rho_c])
    # print(result_root.success)
    # return result_root.x
    result_fsolve = fsolve(func,1.)
    return result_fsolve

def M200_(r200):
    return 4/3*np.pi*r200**3 *200*rho_c

def M_interior_(func, r_up=np.inf): #?? integrate
    ## r_up is r200 or np.inf
    F = integrate.quad( func, 0,r_up ) [0]
    return F

def M_sum_(filename, r200):
    A = np.loadtxt(filename, dtype=float)
    xyz = A[:, 0:3]
    m = A[:,11]
    # r = norm_l(xyz) #??
    # return sum(m*np.sign(r200-r))
    l = len(m)
    M=0
    for i in np.arange(l):
        M += m[i]* isSmaller(norm_l(xyz[i]),r200)
    #     print(i, norm_l(xyz[i]), isSmaller(norm_l(xyz[i]),r200) )
    print("M_sum_:", m[0], norm_l(xyz[0]), M/m[0])
    return M

def dice_r200_(M200):
    return (M200*10.*G*H0)**(1./3)/(10.*H0)

def dice_rho_core_(r):
    return (1+(r/r_s)**2)**(-2.5)

def dice_scale_dens_(cut_R,cut_z):
    val, err = tplquad(lambda z,phi,R : (1. +R**2+z**2)**(-2.5) *R, #func
                0., #R down
                cut_R, #R up
                lambda R : 0., #phi down
                lambda R : 2*np.pi, #phi up
                lambda R,phi : 0., #z down
                lambda R,phi : cut_z) #z up
    print(val)
    return dice_M200/val

def intg_JR(x): #x=kt #all the final index multiplied is k**2/(8*np.pi)
    return np.sin(2*x)**2/(4.-np.sin(x)**2) #this func is one stable binary $\dot R$ of center potential, whose period is np.pi/4
def intg_Jp(x):
    return (4.-np.sin(x)**2)-(1.+7*np.cos(2*x))*2/(4.-np.sin(x)**2)
def intg_Jz(x):
    return 2*np.cos(x)**2

## main
if __name__ == '__main__':
    print(rho_c)

    r = np.linspace(1,10,100)
    rho = rho_Plummer_(r)
    # plt.scatter(r,rho)
    # plt.show()

    #r200 = r200_(foo)
    r200 = r200_(rho_Plummer_solve_) [0]
    dice_r200 = dice_r200_(dice_M200)
    print(r200, dice_r200)

    M200 = M200_(r200)
    print(M200)

    M_interior_r200 = M_interior_(rho_Plummer_, r200*2)
    M_interior_r200_dice = M_interior_(rho_Plummer_, dice_r200*2)
    M_interior_inf  = M_interior_(rho_Plummer_)
    M_interior_2  = M_interior_(rho_Plummer_, 2*1.)
    print(M_interior_r200, M_interior_r200_dice, M_interior_inf, M_interior_2) #(11386030.281301804, 11386666.666666666) ~1.1e7

    filename = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_PM17/txt/snap_0.halo.txt"
    M_sum = M_sum_(filename, r200)
    print(M_sum)

    print(rho_Plummer_(4.394824*r_s))
    print(dice_scale_dens_(dice_r200*2,dice_r200*2))

    JRm_ss = integrate.quad( intg_JR, 0,np.pi/2 ) [0]
    JRm = JRm_ss*10./(8*np.pi)
    Jpm_ss = integrate.quad( intg_Jp, 0,np.pi*2 ) [0]
    Jpm = Jpm_ss*10./(8*np.pi)
    Jzm_ss = integrate.quad( intg_Jz, 0,np.pi ) [0]
    Jzm = Jzm_ss*10./(8*np.pi)
    print("\nactions:")
    print(JRm_ss, JRm)
    print(Jpm_ss, Jpm)
    print(Jzm_ss, Jzm)
    J_all = (JRm**2+Jpm**2+Jzm**2)**0.5
    Jpm_should = (10.**2-JRm**2-Jzm**2)**0.5
    Jpm_ss_should = Jpm_should *8*np.pi/10.
    print(J_all,Jpm_should,Jpm_ss_should)