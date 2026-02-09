#!/usr/bin/env python
# -*- coding:utf-8 -*-

# https://www.cnblogs.com/yymn/p/4716107.html

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import emcee
from numpy.ma.core import maximum_fill_value
from scipy.optimize import curve_fit

# from matplotlib.colors import BoundaryNorm
# import pdb

import galaxy_models
import analysis_data_distribution



#### some settings
C_min = 0.1
C_max = 10.
rater0_min = 0.5
rater0_max = 2.0
raterho0_min = 0.5
raterho0_max = 2.0
alpha_min = 0.01
alpha_max = 3.0
beta_min  = 0.5
beta_max  = 30.0
log_f_min = -200.0
log_f_max = 2.0
# C_min = 0.
# C_max = np.inf
# alpha_min = 0.2
# alpha_max = 2.0
# beta_min = 1.5
# beta_max = 5.0
# log_f_min = -100.0
# log_f_max = 1.0
# alpha<beta
# C_min = 0.1
# C_max = 10.
# rater0_min = 1e-2
# rater0_max = 1e+2
# raterho0_min = 1e-2
# raterho0_max = 1e+2
# alpha_min = 0.001
# alpha_max = 3.0
# beta_min  = 0.5
# beta_max  = 30.0
# log_f_min = -200.0
# log_f_max = 2.0

func_fit_rho1 = galaxy_models.rho_spherical_doublepowerlaw_scaled_coef_log #1+3
func_fit_rho = galaxy_models.rho_spherical_doublepowerlaw_scalednot_log #1+4
# func_fit = galaxy_models.pl2_simple_3d
# func_fit = galaxy_models.AA3_spherical_pl2_Posti15_returnfinite
func_fit = galaxy_models.AA3_spherical_pl2_Posti15_returnfinite_log

M0 = galaxy_models.M_total
r0 = galaxy_models.r_scale
rho0 = galaxy_models.rho_s
# rho0 = galaxy_models.M_total/galaxy_models.r_scale**3
J0 = np.sqrt(galaxy_models.G*galaxy_models.M_total*galaxy_models.r_scale) #about 1e4 now



####:: Defination of prior, likelihood and posterior probability; log them to sum.
# def log_prior(theta):
#     alpha, beta, sigma = theta
#     if sigma < 0:
#         return -np.inf  # log(0)
#     else:
#         return -1.5 * np.log(1 + beta ** 2) - np.log(sigma)
# def log_likelihood(theta, x, y):
#     alpha, beta, sigma = theta
#     # print("shapes likelihood: ")
#     # print(alpha)
#     # print(beta)
#     # print(sigma)
#     # print("sadasasdffsdgfgr")
#     # pdb.set_trace()
#     y_model = alpha + beta * x
#     return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + (y - y_model) ** 2 / sigma ** 2)
# def log_posterior(theta, x, y):
#     return log_prior(theta) + log_likelihood(theta, x, y)

## func scaled rho_r
class fit_MCMC:
    a = 1
    func = func_fit

def log_prior02(theta):
    alpha, beta, _rho0, _r0, log_f = theta
    if  alpha_min<alpha<alpha_max and beta_min<beta<beta_max and rater0_min*r0<_r0<rater0_max*r0 \
        and raterho0_min*rho0<_rho0<raterho0_max*rho0 and alpha<beta and log_f_min<log_f<log_f_max:
        return 0.
    else:
        print("prior wrong!")
        # sys.exit(0)
        return -np.inf  # log(0)
        # return -2

def log_likelihood02(theta, x, y):
    alpha, beta, _rho0, _r0, log_f = theta
    model = func_fit_rho(x, alpha,beta,_rho0,_r0)
    yerr = model*0.1 #??
    # yerr = y.std(0)
    # yerr = np.mean(y.std(0))*J0**3
    # yerr = 0.
    # yerr = 1e4
    # yerr = 1e-10
    # print("sample: model and yerr: ", model.std(0), yerr.std(0))
    sigma2 = yerr**2 +model**2*np.exp(2*log_f) #s_n^2
    return -0.5 * np.sum((y-model)**2/sigma2 +np.log(sigma2))

def log_posterior02(theta, x, y):
    print(theta)
    lp = log_prior02(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + log_likelihood02(theta, x, y)

def log_prior01(theta):
    alpha, beta, _rho0, log_f = theta
    # print(theta)
    if  alpha_min<alpha<alpha_max and beta_min<beta<beta_max and C_min<_rho0<C_min and log_f_min<log_f<log_f_max and alpha<beta:
    # if  alpha_min<alpha<alpha_max and beta_min<beta<beta_max and raterho0_min<C0<raterho0_min and log_f_min<log_f<log_f_max and alpha<beta:
        return 0.
    else:
        print("prior wrong1!")
        # sys.exit(0)
        return -np.inf  # log(0)

def log_likelihood01(theta, x, y):
    alpha, beta, _rho0, log_f = theta
    model = func_fit_rho1(x, alpha,beta,_rho0)
    # yerr = model*0.1 #??
    yerr = np.mean(y.std(0))
    # yerr = np.mean(y.std(0))*J0**3
    # yerr = 0.
    # yerr = 1e4
    # yerr = 1e-10
    sigma2 = yerr**2 +model**2*np.exp(2*log_f) #s_n^2
    return -0.5 * np.sum((y-model)**2/sigma2 +np.log(sigma2))

def log_posterior01(theta, x, y):
    print(theta)
    lp = log_prior01(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood01(theta, x, y)

## func1
def log_prior(theta):
    alpha, beta, C, log_f = theta
    # print(theta)
    if  C_min<C<C_max and alpha_min<alpha<alpha_max and beta_min<beta<beta_max and log_f_min<log_f<log_f_max and alpha<beta:
        # return np.log(C)+alpha*np.log(2.)-beta*np.log(2.)
        return 0.
    else:
        print("wrong!")
        # sys.exit(0)
        return -np.inf  # log(0)

def log_likelihood(theta, x, y):
    alpha, beta, C, log_f = theta
    model = func_fit(x, alpha,beta,C)

    yerr = np.mean(y.std(0)) #??
    # yerr = np.mean(y.std(0))*J0**3
    # yerr = 0.
    # yerr = 1e4
    # yerr = 1e-10
    #: \sigma^2 is measurement error while here are standered error
    # return -0.5 * np.sum(np.log(2 * np.pi * sig ** 2) + (y - y_model) ** 2 / sig ** 2)
    # print("shape: ", x.shape, yerr.shape, y.shape, model.shape)
    # print("standered: ", np.mean(y)) #python: 'numpy.float64' object has no attribute 'type'
    sigma2 = yerr**2 +model**2*np.exp(2*log_f) #s_n^2
    return -0.5 * np.sum((y-model)**2/sigma2 +np.log(sigma2))

## func2
def log_prior2(theta):
    alpha, beta, log_f = theta
    # print(theta)
    if  alpha_min<alpha<alpha_max and beta_min<beta<beta_max and log_f_min<log_f<log_f_max and alpha<beta:
        # return np.log(C)+alpha*np.log(2.)-beta*np.log(2.)
        return 0.
    else:
        print("wrong!")
        # sys.exit(0)
        return -np.inf  # log(0)

def log_likelihood2(theta, x, y):
    alpha, beta, log_f = theta
    model = func_fit(x, alpha,beta)

    # yerr = y*0.1
    yerr = y.std(0)
    print(0.1*np.mean(y), y.std(0))
    # yerr = np.mean(y.std(0))*J0**3
    # yerr = 0.
    # yerr = 1e4
    # yerr = 1e-10
    #: \sigma^2 is measurement error while here are standered error
    # return -0.5 * np.sum(np.log(2 * np.pi * sig ** 2) + (y - y_model) ** 2 / sig ** 2)
    # print("shape: ", x.shape, yerr.shape, y.shape, model.shape)
    # print("standered: ", np.mean(y)) #python: 'numpy.float64' object has no attribute 'type'
    sigma2 = yerr**2 +model**2*np.exp(2*log_f) #s_n^2
    return -0.5 * np.sum((y-model)**2/sigma2 +np.log(sigma2))

## func3
def log_prior3(theta):
    alpha, beta = theta
    # print(theta)
    if  alpha_min<alpha<alpha_max and beta_min<beta<beta_max and alpha<beta:
        # return np.log(C)+alpha*np.log(2.)-beta*np.log(2.)
        return 0.
    else:
        return -np.inf  # log(0)

def log_likelihood3(theta, x, y):
    alpha, beta = theta
    model = func_fit(x, alpha,beta)

    yerr = np.mean(y.std(0)) #??
    # yerr = np.mean(y.std(0))*J0**3
    # yerr = 0.
    # yerr = 1e4
    # yerr = 1e-10
    #: \sigma^2 is measurement error while here are standered error
    # return -0.5 * np.sum(np.log(2 * np.pi * sig ** 2) + (y - y_model) ** 2 / sig ** 2)
    # print("shape: ", x.shape, yerr.shape, y.shape, model.shape)
    # print("standered: ", np.mean(y)) #python: 'numpy.float64' object has no attribute 'type'
    sigma2 = yerr**2 #+model**2*np.exp(2*log_f) #s_n^2
    return -0.5 * np.sum((y-model)**2/sigma2 +np.log(sigma2))

def log_posterior(theta, x, y):
    print(theta)
    log_prior_ = log_prior2
    log_likelihood_ = log_likelihood2
    lp = log_prior_(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_(theta, x, y)



if __name__ == '__main__':
    
    ####:: rho
    ##: data xv set
    galbox = "/home/darkgaia/workroom/0prog/gadget/Gadget-2.0.7/"
    model = "galaxy_general/"
    em = [0]
    cl = ["purple", "blue", "green", "orange", "red"]
    col_xxx = 0
    col_vvv = 3
    col_rho = -4
    datamin = 5.e-2
    datamax = 5.e3

    ## DF
    for j_ss in range(len(em)):
        for i_00X in range(1):

            ## read
            j_X00 = em[j_ss]
            snapid = j_X00*100 + i_00X
            middlename = "snaps/aa000/snapshot_%03d" % (snapid)
            # middlename = "snaps/aa_100/snapshot_%03d" % (snapid)
            postname = ".secondhand_002.txt"
            filename = galbox+model+middlename+postname
            dataread = np.array(np.loadtxt(filename, dtype=float))
            # dataread = np.array(np.where(np.isfinite(dataread), dataread, 0.))
            print(filename)



            ## sample
            xdata = dataread[:,col_xxx:col_xxx+3]
            # xdata = np.hstack((dataread[:,col_JJJ:col_JJJ+3], dataread[:,col_Omg:col_Omg+3]))
            ydata = dataread[:,col_rho]*1e0 #func_fit adjust
            # xdata = xdata[0:1000]
            # ydata = ydata[0:1000]

            xdata_eff, xdata_listCondition,xdata_listConditionNot = analysis_data_distribution.screen_boundary(xdata, datamin,datamax)
            # xdata_eff = abs(xdata_eff)
            # xdata_eff_sample = analysis_data_distribution.norm_l(xdata_eff,0,1)/r0
            xdata_eff_sample = analysis_data_distribution.norm_l(xdata_eff,0,1) /r0
            print(np.median(xdata_eff_sample), np.mean(xdata_eff_sample))

            ydata_eff = ydata[xdata_listCondition]
            # ydata_eff = abs(ydata_eff)
            # ydata_eff_sample = np.log(ydata_eff)-np.log(rho0)
            ydata_eff_sample = np.log(ydata_eff) -np.log(rho0)
            
            # popt = np.array([1.,3.,1.]) #all scalednot
            params_MC = np.array([0.51727206, 0.4984701,  0.54555564, 0.53786437]) #all scalednot
            # params_MC = np.array([8.51615927e-02,  3.87538599e+00,  3.17710527e-01, -9.75579178e+01]) #300 C0
            # params_MC = np.array([0.55074561,    2.75813265,    0.1085531,  -105.10376097]) #000 C0
            # sys.exit(0)
            


            # ####:: MCMC
            # # ndim        = 5     #number of parameters in the model
            # ndim        = 4     #number of parameters in the model
            # nwalkers    = 60*2  #50*steprate #number of MCMC walkers
            # nsteps      = 1000  #2000*steprate #number of MCMC steps to take
            # nburn       = 600   #1000*steprate #"burn-in" period to let chains stabilize

            # np.random.seed(200) # set theta near the maximum likelihood, with 
            # starting_guesses = np.random.random((nwalkers, ndim))

            # sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior01, \
            #     args=[xdata_eff_sample, ydata_eff_sample])
            # # sampler = emcee.EnsembleSampler(nwalkers, ndim, lambda _a,_b,_r0,_logf,_x,_y: log_posterior02([_a,_b,_r0,M0,_logf],_x,_y), \
            # #     args=[xdata_eff_sample, ydata_eff_sample])
            # # sampler = emcee.EnsembleSampler(nwalkers, ndim, lambda [_a,_b,_r0,_logf],_x,_y: log_posterior02([_a,_b,_r0,M0,_logf],_x,_y), \
            # #     args=[xdata_eff_sample, ydata_eff_sample])
            # sampler.run_mcmc(starting_guesses, nsteps)
            # print("MCMC ... Done.")

            # sampler.chain
            # print(sampler.chain.shape)
            # emcee_trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T #-1 is to automacally calculation the rest of shape
            # print(xdata_eff_sample.shape, ydata_eff_sample.shape, emcee_trace.shape)

            # params_MC = np.zeros(ndim)
            # for kk in range(ndim):
            #     params_MC[kk] = np.mean(emcee_trace[kk])



            ## curve_fit
            func = func_fit_rho1
            popt, pcov = curve_fit(lambda _r,_a,_b,_rho0: func(_r,_a,_b,_rho0), xdata_eff_sample, ydata_eff_sample, p0=[1.,3.,1.], \
                bounds=([alpha_min,beta_min,raterho0_min],[alpha_max,beta_max,raterho0_max]), maxfev=5000 )

            # func = func_fit_rho
            # popt, pcov = curve_fit(lambda _r,_a,_b,_r0,_rho0: func(_r,_a,_b,_r0,_rho0), xdata_eff_sample, ydata_eff_sample, p0=[1.,3.,r0,rho0], \
            #     bounds=([alpha_min,beta_min,rater0_min*r0,raterho0_min*rho0],[alpha_max,beta_max,rater0_max*r0,raterho0_max*rho0]), maxfev=5000 )



            # # ## plot 3d
            print("MC mean params: ", params_MC)
            # params_MC = np.array([9.70469942e-02,  3.97888028e+00,  6.69587042e+00, -9.92956669e+01])
            print("LS optimize params: ", popt)
            # 1.05541034 3.43370675 2.        
            print("covariances of fit: ", pcov)

            # yfit_should = func(xdata_eff_sample, 1.0,3.0,19.6, rho0)
            # # yfit_should = func(xdata_eff_sample, 1.,3.,19.6, M0)
            # yfit_LS = func(xdata_eff_sample, popt[0], popt[1], popt[2], popt[3])
            # yfit_MC = func(xdata_eff_sample, params_MC[0], params_MC[1], params_MC[2], params_MC[3])

            yfit_should = func(xdata_eff_sample, 1.0,3.0,1.0)
            yfit_LS = func(xdata_eff_sample, popt[0], popt[1], popt[2])
            yfit_MC = func(xdata_eff_sample, params_MC[0], params_MC[1], params_MC[2])

            dist_0          = analysis_data_distribution.norm_l(ydata_eff_sample,ydata_eff_sample)
            dist_fitshould  = analysis_data_distribution.norm_l(ydata_eff_sample,yfit_should)
            dist_fitMC      = analysis_data_distribution.norm_l(ydata_eff_sample,yfit_MC)
            dist_fitLS      = analysis_data_distribution.norm_l(ydata_eff_sample,yfit_LS)
            print("each dist: ", dist_0, dist_fitshould, dist_fitMC, dist_fitLS)

            plt.scatter(xdata_eff_sample, ydata_eff_sample, color="black",s=0.5)
            plt.scatter(xdata_eff_sample, yfit_should, color="red",s=0.5)
            # plt.scatter(xdata_eff_sample, yfit_MC, color="green",s=0.5)
            # plt.scatter(xdata_eff_sample, yfit_LS, color="blue",s=0.5)
            plt.xlabel(r"$r/r_0$", fontsize=20)
            plt.ylabel(r"$\rho/\rho_0$", fontsize=20)
            plt.xscale("log")
            # plt.yscale("log")
            plt.show()





'''
if __name__ == '__main__':

    ####:: J
    ####:: data generating
    ##: example
    # np.random.seed(38)
    # N_data = 80
    # # xdata = 100 * np.random.random(N_data)
    # # xdata = 1.*np.random.random((N_data,3))
    # xdata_1 = 1.*np.linspace(0.1,1., N_data)
    # xdata = np.array([xdata_1,xdata_1,xdata_1]).T
    # # xdata = np.random.normal(xdata, 1.e0) # add scatter to points
    # xdata = abs(xdata)

    # # theta_true = np.array([25, 0.5])
    # # ydata0 = theta_true[0] + theta_true[1] * (xdata[:,0]+xdata[:,1]+xdata[:,2])
    # ydata0 = galaxy_models.pl2_simple_3d(xdata, 1.,3.,10.)
    # ydata = np.random.normal(ydata0, 1.e0)
    # ydata = abs(ydata)
    # print("data: ", xdata, ydata0)
    # print( np.mean(xdata.std(0)), np.mean(ydata.std(0)) )

    ##: data actions
    ## f1, f2, f3 of one halo
    galbox = "/home/darkgaia/workroom/0prog/gadget/Gadget-2.0.7/"
    # galbox = "/home/darkgaia/0prog/gadget/gadget-2.0.7/"
    # galbox = "/home/darkgaia/0prog/gadget/gadget-2.0.7/galsbox/"
    # galbox = "/home/darkgaia/0prog/gadget/gadget-2.0.7/galsbox/gals_20210903_actions_spherical_1e4_centered/"
    model = "galaxy_general/"
    # model = "galaxy_general_1_NFW"
    # model = "galaxy_general_2_Hernquist"
    # model = "galaxy_general_3_Burkert"
    # model = "galaxy_general_4_Einasto"
    # model = "galaxy_general_5_isothermal"
    # model = "galaxy_general_6_Plummer"
    ## snapshot/100
    em = [0]
    # em = [0, 1, 2, 3]
    ## under coordinates: spherical: 12 18, 45 51; axisymmetric: 21 27, 54 60; triaxial: 30 37, 63 70
    col_JJJ = 9
    col_Omg = 6
    col_fJ = -3
    ## others
    cl = ["purple", "blue", "green", "orange", "red"]
    k_knn = 10
    # datamin = 1.e-10 #1.e-8
    # datamax = 1.e+10  #1.e4
    datamin = 2.e-2
    datamax = 2.e4

    ## DF
    for j_ss in range(len(em)):
        for i_00X in range(1):
            j_X00 = em[j_ss]
            snapid = j_X00*100 + i_00X
            middlename = "snaps/aa000/snapshot_%03d" % (snapid)
            postname = ".secondhand_005.txt"
            filename = galbox+model+middlename+postname
            dataread = np.array(np.loadtxt(filename, dtype=float))
            # dataread = np.array(np.where(np.isfinite(dataread), dataread, 0.))
            print(filename)



            ## fJ bins
            # x_points_inaline,fx_points_inaline, xx_points_inaline,fxx_points_inaline, xxx_points_inaline,fxxx_points_inaline, D1_datapoints,D2_datapoints \
            #     = analysis_data_distribution.divided_bins_123(dataread, colD=col_JJJ,colD2=col_Omg, \
            #     nmesh=100,nmesh2=10,nmesh3=6, whatbin=1, datamin=datamin,datamax=datamax) #whatbin=1,4

            # x_cellpoints = xxx_points_inaline #the name: datapoints and mesh??
            # Omg_cellpoints = analysis_data_distribution.fromAAAtoBBB_byinterpolation_atsomepoints(D1_datapoints,D2_datapoints,x_cellpoints,k=k_knn,funcname="gaussian") #interpolation
            # Omg_cellpoints = np.array( np.where(Omg_cellpoints==0., datamin, Omg_cellpoints) ) #remove zero
            # x_cellpoints_JOmg = np.array( np.hstack((x_cellpoints, Omg_cellpoints)) ) #horizontal merging to be as input argument
            # y_cellpoints = fxxx_points_inaline
            # # func = galaxy_models.AA3_spherical_pl2_Posti15_interpolation
            # # popt, pcov = curve_fit(func, x_cellpoints_JOmg, y_cellpoints, p0=[1.,3., 1e-2]) #for frequencies



            ## fJ sph kernel
            ## remove 0, bin bad value, bin, fit rho, corner, fit func and index ??
            # xdata = x_cellpoints_JOmg
            # ydata = y_cellpoints
            xdata = np.hstack((dataread[:,col_JJJ:col_JJJ+3], dataread[:,col_Omg:col_Omg+3]))
            ydata = dataread[:,col_fJ]*1e0 #func_fit adjust
            # xdata = xdata[0:1000]
            # ydata = ydata[0:1000]
            xdata_eff, xdata_listCondition,xdata_listConditionNot = analysis_data_distribution.screen_boundary(xdata, datamin,datamax)
            xdata_eff[:,1] = abs(xdata_eff[:,1]) #Lz
            xdata_eff = abs(xdata_eff)
            ydata_eff = ydata[xdata_listCondition]
            ydata_eff_log = np.log(ydata_eff*J0**3)
            
            # J_compose1 = (xdata[:,0]*xdata[:,3] +xdata[:,1]*xdata[:,4] +xdata[:,2]*xdata[:,5])
            # # print(J_compose1)
            # # print(xdata[:,3])
            # # whereis = np.array(np.where(abs(xdata[:,3])<1e-10))
            # whereis = np.array(np.where(abs(xdata[:,3])<1e-10 or abs(xdata[:,4])<1e10))
            # J_compose1 /= (xdata[:,3]*J0)
            # # print(len(whereis), sum(whereis))
            # # print(J_compose1)
            # print(whereis, whereis.shape)
            # print(J_compose1[whereis])
            # xdata_eff = np.array(np.where(np.isfinite(xdata_eff),xdata_eff,0.))
            # print(xdata[xdata_listConditionNot, 0])
            # print(xdata_eff.shape)

            J = xdata_eff[:,0:3]
            Omg = xdata_eff[:,3:6]
            h = J[:,0]*Omg[:,0] +J[:,1]*Omg[:,1] +J[:,2]*Omg[:,2]
            h = abs(h/Omg[:,0]) #pm
            g = J[:,0] +J[:,1] +J[:,2]
            J0_here = J0
            C = 1. #J0**3/J0**3
            k1 = 1.
            k2 = 3.
            power1 = (6.-k1)/(4.-k1)
            power2 = (2.*k2-3.)
            base1 = 1.+J0_here/h
            base2 = 1.+g/J0_here
            rtn = np.log(C) +power1*np.log(base1) -power2*np.log(base2)
            rtn_finite = np.array(np.where(np.isfinite(rtn), rtn,0.))
            print(np.median(h), np.median(g), np.median(base1), np.median(base2))
            # sys.exit(0)
            


            ####:: MCMC
            # Here we'll set up the computation. emcee combines multiple "walkers",
            # each of which is its own MCMC chain. The number of trace results will
            # be nwalkers * nsteps
            # sampler.chain is of shape (nwalkers, nsteps, ndim)
            # we'll throw-out the burn-in points and reshape

            # ndim = 2  # number of parameters in the model
            ndim     = 3
            steprate = 1
            nwalkers = 60*2 #50*steprate # number of MCMC walkers
            nsteps   = 1000 #2000*steprate # number of MCMC steps to take
            nburn    = 600 #1000*steprate # "burn-in" period to let chains stabilize

            np.random.seed(20) # set theta near the maximum likelihood, with 
            starting_guesses = np.random.random((nwalkers, ndim))

            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[xdata_eff, ydata_eff_log])
            sampler.run_mcmc(starting_guesses, nsteps)
            print("Walking ... Done.")

            sampler.chain
            print(sampler.chain.shape)
            emcee_trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T #-1 is to automacally calculation the rest of shape
            print(xdata.shape, ydata.shape, emcee_trace.shape)
            # print("data: ", xdata, ydata)
            print( np.mean(xdata_eff.std(0)), np.mean(ydata_eff.std(0)) )
            print("scale:", M0,J0,M0/J0**3)

            params_MC = np.zeros(ndim)
            for kk in range(ndim):
                params_MC[kk] = np.mean(emcee_trace[kk])
            print("MC mean params: ", params_MC)



            ## curve_fit
            func = func_fit
            # popt, pcov = curve_fit(func, x_cellpoints_JOmg, y_cellpoints, p0=[1.,3., 1e-2], bounds=([alpha_min,alpha_max],[beta_min,beta_max],[C_min,C_max]))
            popt, pcov = curve_fit(func, xdata_eff, ydata_eff, p0=[1.,3.], bounds=([0.,2.],[2.,5.]))
            print("LS optimize params: ", popt)
            print("covariances of fitting: ", pcov)



            # ## plot 3d
            x_compose1 = ( xdata_eff[:,0]**2 +xdata_eff[:,1]**2 +xdata_eff[:,2]**2 )**0.5/J0
            x_compose2 = h/J0
            x_compose3 = g/J0
            x_compose4 = rtn
            # params_MC = np.array([0.94810531, 1.93365828, -34.59636356]) #narrow range of beta
            # params_MC = np.array([1.0915327, 2.62831516, -104.22685548]) #wide range of beta
            # params_MC = [1.02608027    2.52427954 -100.65931372]
            x_compose = x_compose4
            # yfit = func_fit(xdata_eff, 0.05, 3.2)
            yfit_should = func_fit(xdata_eff, 1.,3.)
            yfit_MC = func_fit(xdata_eff, params_MC[0], params_MC[1])
            yfit_LS = func_fit(xdata_eff, popt[0], popt[1])
            # yfit_MC = func_fit(xdata_eff, 1.0915327, 2.62831516)
            # print("each min: ", min(ydata_eff_log), min(yfit_should), min(yfit_LS), min(yfit_MC))

            dist_0          = analysis_data_distribution.norm_l(ydata_eff_log,ydata_eff_log )
            dist_fitshould  = analysis_data_distribution.norm_l(ydata_eff_log,yfit_should   )
            dist_fitLS      = analysis_data_distribution.norm_l(ydata_eff_log,yfit_LS       )
            dist_fitMC      = analysis_data_distribution.norm_l(ydata_eff_log,yfit_MC       )
            print("each dist: ", dist_0, dist_fitshould, dist_fitMC, dist_fitLS)

            # plt.scatter(x_compose, ydata_eff_log/x_compose,   color="black",s=0.5)
            # plt.scatter(x_compose, yfit_should/x_compose,     color="red",s=0.5)
            # plt.scatter(x_compose, yfit_LS/x_compose,         color="green",s=0.5)
            # plt.scatter(x_compose, yfit_MC/x_compose,            color="blue",s=0.5)

            plt.scatter(x_compose, ydata_eff_log/1,   color="black",s=0.5)
            plt.scatter(x_compose, yfit_should/1,     color="red",s=0.5)
            plt.scatter(x_compose, yfit_LS/1,         color="green",s=0.5)
            plt.scatter(x_compose, yfit_MC/1,            color="blue",s=0.5)
            plt.xscale("log")
            plt.yscale("log")
            plt.show()





            ## bins
            # yfit = func_fit(x_cellpoints_JOmg, *params_MC[0:3])
            # yfit2 = func_fit(x_cellpoints_JOmg, *popt)
            # nl = len(fxxx_points_inaline)
            # print(xdata)
            # # print(ydata, yfit)

            # ftsz = 24
            # cut = 4
            # xlist = np.arange(nl)
            # yfit_tgt = func_fit(x_cellpoints_JOmg, 1.,3.,3.)
            # plt.plot(xlist,yfit,label="MC")
            # # plt.plot(xlist,yfit2,label="LS")
            # # plt.plot(xlist,yfit_tgt,label="fitting target")
            # plt.plot(xlist,fxxx_points_inaline,label="original data")
            # plt.legend(fontsize=ftsz)
            # plt.show()

            # fig = plt.figure(figsize=(16, 12), dpi=80, facecolor=(0.0, 0.0, 0.0))
            # ax = Axes3D(fig)
            # ax.grid(True) # ax.set_axis_off() #or to remove all relevent axis
            # ax.scatter(x_cellpoints[:,0], x_cellpoints[:,1], y_cellpoints[:], color="red", s=1.5, label="samples (%d points)"%(nl))
            # ax.scatter(x_cellpoints[:,0], x_cellpoints[:,1], yfit[:], color="blue", s=1.5, label="fitted (%d points)"%(nl))
            # ax.plot([x_cellpoints[0,0], x_cellpoints[0,0]], [x_cellpoints[0,1], x_cellpoints[0,1]], [y_cellpoints[0], yfit[0]], color="green", lw=0.5, \
            #     label="differences between by data points and fit value") #+'\n'+'(only 1/5 of these differences are displayed for clarity)
            # for i in np.arange(len(x_cellpoints[:,0])): # (7000,8000): 
            #     ax.plot([x_cellpoints[i,0], x_cellpoints[i,0]], [x_cellpoints[i,1], x_cellpoints[i,1]], [y_cellpoints[i], yfit[i]], color="green", lw=0.5)
            # ax.set_xlim(0.,1000.)
            # ax.set_ylim(0.,2000.)
            # ax.set_xlabel(r'r-action $J_r\quad \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=ftsz)
            # ax.set_ylabel(r'$\phi$-action $L_z\quad \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=ftsz)
            # ax.set_zlabel(r'distribution function (divided by bin) $f(x_1,x_2,x_3)\, \times 1$'+'\n'+r'(here \
            #     only $x_1$ and $x_2$ axis displayed, $x_3$ axis are overlapped)', fontsize=ftsz)
            # ax.legend(fontsize=ftsz)
            # ax.view_init(elev = 0., azim = 180.)
            # plt.show()
'''
