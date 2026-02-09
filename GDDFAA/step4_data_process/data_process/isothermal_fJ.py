#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import f_J__plot

G = 43007.1 #(10^-10*M_\odot, kpc, km/s)
#Sun = np.array([8.178, 0.1, 0.225,    25.13, 180.01, 14.95]) #the Sun xv
R0 = 8.2 #kpc; the sun in MW
v0 = 220.
L0 = R0*v0 #kpc*km/s; the sun in MW
#Rg = Rc #kpc; Guiding-Center Radius; ??                       one, by potential

sigma_R_0 = 66. #km/s                                          one, to fit or from sigma_R(R0) = 66 km/s, so it is 66 km/s
sigma_z_0 = 55. #km/s                                          one, ... z 55 km/s
h_R = 2. #kpc
h_sigma__R = 7. #kpc
h_sigma__z = 8. #kpc

def phi_r(r):

    ## params
    rho0 = 0.000854 #e10*M_\odot kpc^-3
    r0 = 19.6 #kpc
    Ms = 4.*np.pi *rho0 *r0**3
    rr = r/r0
    print(Ms)
    ## model expression
    # ## power law:
    # alpha = 1.53
    # rho = rho0 *rr**alpha
    # phi = -G*Ms* r**(-alpha)
    ## NFW:
    rho = rho0 / ( rr**1 *(1+rr)**(3-1) )
    phi = -G*Ms *np.log(1.+rr)/r
    return phi

def norm_l(a,l=2):
    x=0
    for e in a:
        x += e**l
    return x**(1./l)

def Rc__(Lz): #circular radius
    return 0.
def vc__(Rc): #circular velocity
    return 0.

def sigma_R(Rc): #velocity dispersion
    return sigma_R_0*np.exp( -(Rc-R0) / h_sigma__R )
def sigma_z(Rc):
    return sigma_z_0*np.exp( -(Rc-R0) / h_sigma__z )
def nRc(Rc): #surface density (J Read, 2014) ??
    #return 1.*np.exp( -Rc / h_R )
    return 0.0069*np.exp( -(Rc-R0) / 2.5 ) #(10^-10*M_\odot, kpc, km/s)

# def Omega(Rc, Lz): #frequency
#     return Lz/Rc**2
# def kappa(Rc, Lz): #frequency
#     return 1.
# def nu(Rc, Lz): #frequency
#     return 1.

def f(JR, Lz, Jz,    Rc,kappa,Omega,nu): #the long isothermal DF equation # bad prog, intermedia calculation is by reading data
    f_sig__R = Omega*nRc(Rc)/(np.pi*kappa*sigma_R(Rc)**2) *(1+np.tanh(Lz/L0)) *np.exp(-kappa*JR/sigma_R(Rc)**2)
    f_pz = nu/(2*np.pi*sigma_z(Rc)**2) *np.exp(-nu*Jz/sigma_z(Rc)**2)
    f_qDF = f_sig__R*f_pz
    return f_qDF

def f_Lz(Lz,    Rc,kappa,Omega,nu): #\int_0^inf f(J_R,L_z,J_z) dJ_R dJ_z
    fLz = Omega*nRc(Rc)/(kappa**2) *(1.+np.tanh(Lz/L0))
    return fLz

# def f_JRLz(Lz,    Rc,kappa,Omega,nu): #\int_0^inf f(J_R,L_z,J_z) dJ_z
#     fLz = Omega*nRc(Rc)/(kappa**2) *(1.+np.tanh(Lz/L0))
#     return fLz



if __name__ == '__main__':

    ## load
    # ##xvp
    # ICfile = "/home/darkgaia/0prog/gadget/gadget-2.0.7/galaxy_Bovy13/snaps/snapshot_000.txt"
    # IC = np.loadtxt(ICfile)
    # x = IC[:, 0:3]
    # v = IC[:, 3:6]
    # phi = IC[:, 13]

    ## f_J
    # filename = "/home/darkgaia/0prog/data_process/aa/20210314_Bovy13/doublepl_1e4_all.qDF"
    # filename = "/home/darkgaia/0prog/data_process/aa/20210314_Bovy13/doublepl_1.8e5_Lzlist.qDF"
    # filename = "/home/darkgaia/0prog/data_process/aa/20210314_Bovy13/Bovy13_plO_1.8e5.formuladata"
    # filename = "/home/darkgaia/0prog/data_process/aa/20210314_Bovy13/Bovy13_plO_1.8e5.formuladata"
    # filename = "/home/darkgaia/0prog/data_process/aa/20210314_Bovy13/Bovy13_pl_1.8e6.formuladata"
    # filename = "/home/darkgaia/0prog/data_process/aa/20210314_Bovy13/Bovy13_0_pl1.formuladata"
    # filename = "/home/darkgaia/0prog/data_process/aa/20210314_Bovy13/Bovy13_0.formuladata"
    filename = "/home/darkgaia/0prog/data_process/aa/20210314_Bovy13/PlummerBP_1e4_all.qDF"
    data = np.loadtxt(filename) #select #gas 0~2000, halo 2000~12000, disk 12000~17000, bulge 17000~18000 #ASF FP 0:3, TSF FP 3:6, ASF DP 7:10, TSF DP 10:13
    nn = data[:,0]
    l = len(nn)
    Lz = data[:,1]

    Rc = data[:,2]
    vc = data[:,3]
    kappa = data[:,4]
    Omega = data[:,5]
    nu    = data[:,6]
    RR = data[:,7]
    P = data[:,8]
    Fn = data[:,12]
    Pxx = data[:,13]
    Pxx_ds = data[:,36]
    sig = data[:,42]

    Rc0 = data[:,16]
    vc0 = data[:,17]
    kappa0 = data[:,18]
    Omega0 = data[:,19]
    nu0    = data[:,20]
    RR0 = data[:,21]
    P0 = data[:,22]
    Fn0 = data[:,26]
    Pxx0 = data[:,27]
    Pxx_ds0 = data[:,39]
    sig0 = sigma_R(RR0)



    # kappa = data[:,30] #ds
    # Omega = data[:,31]
    # nu    = data[:,32]
    P_pl = phi_r(RR0) #-1.53*4e7* RR0**-1.53

    JR = 10.
    Jz = 10.
    fJ = f(JR,Lz,Jz,    Rc,kappa,Omega,nu)
    fLz = f_Lz(Lz,    Rc,kappa,Omega,nu)
    fJ0 = f(JR,Lz,Jz,    Rc0,kappa0,Omega0,nu0)
    fLz0 = f_Lz(Lz,    Rc0,kappa0,Omega0,nu0)
    print(fJ)
    print(fLz)
    print(phi_r(1e-10))



    ## plot
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"] #\boldsymbol

    # plt.plot(RR[1:], Rc[1:], label="Data potential system") #Rg
    # plt.plot(RR0[1:], Rc0[1:], label="Formula potential system")
    # plt.plot(RR[1:], kappa[1:], label="Data potential system") #three freqs are all smaller, kappa osc
    # plt.plot(RR0[1:], kappa0[1:], label="Formula potential system")

    plt.plot(RR[:], P[:], label="Data potential system") #P
    plt.plot(RR0[:], P0[:], label="Formula potential system")
    plt.plot(RR0[1:], P_pl[1:], label="powerlaw potential system")
    # plt.plot(RR[1:], Fn[1:], label="Data potential system") #Fn
    # plt.plot(RR0[1:], Fn0[1:], label="Formula potential system")
    # plt.plot(RR [:], Pxx   [:], label="Data potential system 3pointsdifference") #PXX_x
    # plt.plot(RR [:], Pxx_ds[:], label="Data potential system directsummation")
    # plt.plot(RR0[:], Pxx0  [:], label="Formula potential system 3pointsdifference")
    # plt.plot(RR[:], sig[:], label="Data potential system") #sig_diag
    # plt.plot(RR0[:], sig0[:], label="Formula potential system")

    # plt.plot(Lz, fLz, label="Data potential") #fLz
    # plt.plot(Lz, fLz0, label="Formula potential")
    # plt.plot(Lz, fJ, label="Data potential") #fqDF_atXX
    # plt.plot(Lz, fJ0, label="Formula potential")
    plt.xlabel(r'R in x-axis $x\quad (\mathrm{kpc})$', fontsize=14) #M_\odot
    plt.ylabel(r'$...$', fontsize=14)
    # plt.xlabel(r'$L_z\quad (\mathrm{kpc}\cdot\mathrm{km}\cdot s^{-1})$', fontsize=14) #M_\odot
    # plt.ylabel(r'quasi-isothermal $f(J_R,Lz,Jz)|_{(J_R=Jz=10\mathrm{kpc}\cdot\mathrm{km}\cdot s^{-1})}$', fontsize=14)
    plt.legend()
    plt.savefig(filename+".png", dpi=200)
    plt.show()

    # plt.plot(RR, p)
    # plt.show()
    ## load2 compare