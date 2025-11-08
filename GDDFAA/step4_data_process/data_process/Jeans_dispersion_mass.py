#!/usr/bin/env python
# -*- coding:utf-8 -*-

#### to get Jeans mass of a given galactic model ####

import numpy as np
import matplotlib.pyplot as plt
import sympy
from scipy import integrate

##constants:
G = 6.67e-11 #units:International(m,s,kg) #gravatational constant
onekpc = 3.0857e19 #units:International
M_sun = 1.99e30 #units:International
v_sun = 2.2e5 #units:International

rho0 = 0.00854*M_sun/onekpc**3 #units:International #scale mass density
a = 19.6*onekpc #units:kpc #scale length
# x:=r/a, r=a*x, a=r/x
Mg_this = 2*np.pi*a**3*1e10 #mass of outer matter of galaxy
Mg_that = 1.6e12*M_sun #units:International
Mg_this = 2*np.pi*rho0*a**3
Mg = Mg_this
m_passive = 1. #no influrence if divided at last

Mbh__Mg_default = 0.002 #ratio of mass of central black hole and Mg
beta_sigma_default = 0. #anisotropy param

##models:
class NFW:
    def __init__(self,bs,mm):
        self.alpha=1 #powers of power-law mass density model, for NFW alpha=1 and beta=3
        self.beta=3
        beta_sigma=bs
        Mbh__Mg=mm
        self.f0=lambda xx: (xx*a)**(2*beta_sigma)  *1./Mg *1./(xx**self.alpha * (1+xx)**(self.beta-self.alpha))  **m_passive*( G*Mbh__Mg*Mg/(xx*a)**2 -4*np.pi*G*rho0*a/xx**2*( xx/(1+xx)-np.log(1+xx) ))
        # F = integrate.quad( f1*f2*f3, r,np.inf ) #error: unsupported operand type(s) for *: 'function' and 'function'
        # r->x, all need to change include up limit and dowm limit; the original d rr in [r,inf] should be d xx in [a*x,inf] *a

class Hernquest:
    def __init__(self,bs,mm):
        self.alpha=1
        self.beta=4
        beta_sigma=bs
        Mbh__Mg=mm
        self.f0=lambda xx: (xx*a)**(2*beta_sigma)  *1./Mg *1./(xx**self.alpha * (1+xx)**(self.beta-self.alpha))  *m_passive*( G*Mbh__Mg*Mg/(xx*a)**2 -4*np.pi*G*rho0*a**2*( -1./(2*a)*1./(1+xx)**2) )

def M_Jeans__plot(model):
    if model!=1 and model!=2:
        print("No such model!")
        return 0
    bs_ = [0.0, -0.5,  0.5]
    mm_ = [0.0]#,0.002,0.004]
    n = 100
    x0 = 0.001
    x_ = np.linspace(x0,10.,n)
    x_ = np.logspace(-2.,1.,100)
    sigf_ = np.zeros(n)
    sigfs_ = np.zeros(n)
    sigs_ = np.zeros(n)
    sig_ = np.zeros(n)
    sig_sig_ = np.zeros(n)
    MJ_ = np.zeros(n) #don't MJ_=vs2_, they will be same all day
    plt.figure()
    for Mbh__Mg in mm_:
        for beta_sigma in bs_:
            nfw = NFW(beta_sigma,Mbh__Mg) #model 1
            hernq = Hernquest(beta_sigma,Mbh__Mg) #model 2
            list_model = [nfw,hernq]
            alpha = list_model[model-1].alpha
            beta = list_model[model-1].beta
            f0 = list_model[model-1].f0

            j=0
            for x in x_:
                F = integrate.quad( f0, x,np.inf ) #this attribute has two values!!
                nu = 1./Mg *1./(x**alpha * (1+x)**(beta-alpha))
                vs2 = 1./((a*x)**(2*beta_sigma)*nu) *F[0]*a

                ##Binney page351 vs2
                # ff = lambda xx: xx**(2*beta_sigma-1)/(1+xx)**5 +Mbh__Mg*xx**(2*beta_sigma-3)/(1+xx)**3
                # F = integrate.quad( ff, x,np.inf )
                # vs2 = G*Mg/a *(1+x)**3/x**(2*beta_sigma-1) *F[0]
                ##Binney page351 sig when beta_sigma=0.
                vva_gm = (1+6*Mbh__Mg)*x*(1+x)**3*np.log(1/x+1)-Mbh__Mg*(1+x)**3*(3*x-0.5)/x-x*( 0.25+1./3*(1+x)+(1+Mbh__Mg)/2*(1+x)**2+(1+3*Mbh__Mg)*(1+x)**3 )/(1+x) #here beta_sigma=0.
                sigf_[j] = np.sqrt(vva_gm)
                sigfs_[j] = np.sqrt(vva_gm)  *np.sqrt(G*Mg_that*m_passive/a)

                sig_[j] = np.sqrt(vs2) #its no use because unknow Mg~M0, its better to use dimless sig then =/sqrt(G*M_thatMW*m_passive/a)
                sig_nodim = np.sqrt(vs2) /np.sqrt(G*Mg*m_passive/a)
                MJ_nodim = 2.92 *G**(-1.5)*rho0**(-0.5) *vs2**1.5 /Mg #1./6*np.pi**1.5
                sig_sig_[j] = np.sqrt(vs2) /np.sqrt(G*Mg*m_passive/a)
                sigs_[j] = sig_nodim  *np.sqrt(G*Mg_that*m_passive/a)
                MJ_[j] = MJ_nodim
                j += 1
            print(sig_[0],sig_sig_[0],sigs_[0],sigf_[0],sigfs_[0],MJ_[0])
            plt.axes(xscale = "log") #should be at front of plt.plot()

            # plt.plot(x_,sig_,label=r"$\beta_\sigma=%f, M_\mathrm{BH}=%f$"%(beta_sigma,Mbh__Mg))
            # plt.plot(x_,sig_sig_,label=r"$\beta_\sigma=%f, M_\mathrm{BH}=%f$"%(beta_sigma,Mbh__Mg))
            plt.plot(x_,sigs_,label=r"$\beta_\sigma=%f, M_\mathrm{BH}=%f$"%(beta_sigma,Mbh__Mg)) #
            # plt.plot(x_,sigf_,label=r"$\beta_\sigma=%f, M_\mathrm{BH}=%f$"%(beta_sigma,Mbh__Mg))
            # plt.plot(x_,sigfs_,label=r"$\beta_\sigma=%f, M_\mathrm{BH}=%f$"%(beta_sigma,Mbh__Mg))
            # plt.plot(x_,MJ_,label=r"$\beta_\sigma=%f, M_\mathrm{BH}=%f$"%(beta_sigma,Mbh__Mg)) #

    plt.legend()
    plt.xlabel(r"dimensionless radius  $r/a$",size=20.)
    # plt.ylabel(r"radical velocity diepersion $\sigma_{rr}(\mathrm{m/s})$, PM07 MW paras: M_g, r_h",size=20.)
    # plt.ylabel(r"radical velocity diepersion  $\sigma_{rr}/\sqrt{2G\pi\rho0 a^2}$(dimensionless)",size=20.)
    plt.ylabel(r"Jeans mass $M_\mathrm{Jeans}/2\pi\rho_0 r_h^3$(dimensionless)",size=20.)
    # plt.title(r"Hernquest model with a central BH",size=20.) #Hernq
    plt.title(r"NFW model with a central BH",size=20.) #Hernquest
    plt.show()
    return 1

if __name__ == '__main__':
    M_Jeans__plot(1)
    print(Mg_that,Mg_this,Mg)
    print(np.sqrt(G*Mg*m_passive/a))
    ##when a=8.8kpc, sig_noodim=0.32
    ##when a=8.8kpc, sig_yesdim=1.92e5 m/s, by MW: a and M
    print(np.log(2)/np.log(10)) #=0.301
    print(np.log(8.2/19.6)/np.log(10)) #=0.301



####discard:
# r,r1,r2,r3 = sympy.symbols("r,r1,r2,r3")

# def rho_2powerlaw(alpha=1,beta=3): # r, rho0,a, 
#     return rho0/( (r/a)**alpha * (1+r/a)**(beta-alpha) )

# def foo(): #test
#     return sympy.exp(-a*r)
# def M_tot(alpha=1,beta=3):
#     # f1 = foo()
#     # return sympy.integrate(f1.subs(a,2), (r,1,np.inf))
#     # f1 = rho_2powerlaw()
#     # return sympy.integrate(f1.subs(rho0,1).subs(a,1), (r,0,np.inf))
#     return 2*np.pi*a**3*rho0

# def f_xv(x,v):
#     return 0. #which f-model for NFW??
# def nu_2powerlaw(alpha=1,beta=3): # r, rho0,a, #rho=nu??
#     return 1./(2*np.pi*a**3) *1./( (r2/a)**alpha * (1+r2/a)**(beta-alpha) )

# def dPhi_r():
#     return -4*np.pi*G*rho0*a**2 *( r3/a**2/(1+r3/a)-1./a*sympy.log(1+r3/a) )/(r3**2/a**2)

# def sigma2_rr_Jeans(beta_sigma=0.): 
#     # vr_bar =  integ vr f_xv
#     # rho =  integ vvv f_xv * integ xxx rho
#     # nu =  integ vvv f_xv
#     # sig2 =  integ vvv (vr-vr_bar)**2*f_xv /nu
#     nu = nu_2powerlaw()
#     dPhi_r1 = dPhi_r()
#     #*dPhi_r1.subs(r3,r1) #fucking python sympy, inner first, so no spesific value
#     #fucking python cannot make sympy.log() as integratedor??
#     return 1./(r**(2*beta_sigma)*nu.subs(r2,r)) *sympy.integrate( r1**(2*beta_sigma)*nu.subs(r2,r1) *(-4*np.pi*G*rho0*a**2 *( r1/a**2/(1+r1/a)-1./a*sympy.log(1+r1/a) )/(r1**2/a**2)), (r1, r,np.inf) )

# def M_Jeans(lambda_Jeans, rr, opt=0):
#     if opt==0:
#         return 4*np.pi/3 *G*rho0* (lambda_Jeans/2)**3 #def
#     if opt==1:
#         v_s = sigma2_rr_Jeans()
#         return 1./6*np.pi**1.5 *G**(-1.5)*rho0**(-0.5) *v_s.subs(r,rr)**3

# if __name__ == '__main__':

#     r_in = np.linspace(0.1,10.,10)
#     MJ = r_in
#     for i in range(10):
#         #MJ[i] = M_Jeans(1,r_in[i],1)
#         print(M_Jeans(1,r_in[i],1))
#     MJ = MJ
#     plt.plot(r_in,MJ)
#     plt.show()