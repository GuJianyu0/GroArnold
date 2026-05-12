#!/usr/bin/env python
# -*- coding:utf-8 -*-

# ===================================================================================
# Author: Jianyu Gu
# Description: To fit galaxy models of mass density and action probability density.
# ===================================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import emcee
import corner
from numpy.core.fromnumeric import shape  # >= python3, >= ubuntu 20.0
import scipy.optimize as spopt
from scipy.optimize import curve_fit
import pdb
from tqdm import tqdm
import os
import sys
# from IPyhton.display import Latex

import time
from multiprocessing import Pool
from multiprocessing import cpu_count
# from schwimmbad import MPIPool

import galaxy_models as gm
import analysis_data_distribution as add



Dim = 3
colors = ["red", "orange", "olive", "green", "cyan", "blue", "purple", "pink", "gray", "black"]

####[] the params setting of galaxy model
class Model_galaxy:
    '''
    The galaxy model. Some of the params should be given while other params should be fit.
    '''

    def __init__(self, M=1., ls=1., ds=1.):
        # params: read value, reference value, down value and up value
        self.params_dict = {
            'mass'              : np.ones(4)*M,  # start
            'length_scale'      : np.ones(4)*ls,
            'density_scale'     : np.ones(4)*ds,
            'action_scale'      : np.ones(4)*np.sqrt(gm.G*M*ls),
            "scale_free_1"      : np.ones(4), 
            "scale_free_2"      : np.ones(4), 
            "scale_free_3"      : np.ones(4), 
            "scale_free_4"      : np.ones(4), 
            
            'axis_ratio_x'      : np.ones(4),
            'axis_ratio_y'      : np.ones(4),
            'axis_ratio_z'      : np.ones(4),
            'rotate_angle_x'    : np.ones(4),
            'rotate_angle_y'    : np.ones(4),
            'rotate_angle_z'    : np.ones(4),

            'power_alpha'       : np.ones(4), #double power-law gal
            'power_beta'        : np.ones(4), #double power-law gal
            'power_Einasto'     : np.ones(4), #Einasto gal
            'log_penalty'       : np.ones(4),
            
            'coef_total'        : np.ones(4),
            "coef_free_n3"      : np.ones(4),
            "coef_free_n2"      : np.ones(4),
            "coef_free_n1"      : np.ones(4),
            "coef_free_0"       : np.ones(4),
            "coef_free_1"       : np.ones(4),
            "coef_free_2"       : np.ones(4),
            "coef_free_3"       : np.ones(4),
            "coef_free_4"       : np.ones(4),
            "coef_free_5"       : np.ones(4),
            "coef_free_6"       : np.ones(4),
            "coef_free_7"       : np.ones(4),
            "coef_free_8"       : np.ones(4),
            "coef_free_9"       : np.ones(4),
            "coef_free_p0"      : np.ones(4),
            "coef_free_p1"      : np.ones(4),
            "coef_free_p2"      : np.ones(4),
            "coef_free_p3"      : np.ones(4),
            "coef_free_p4"      : np.ones(4),
            "coef_free_p5"      : np.ones(4),
            "coef_free_p6"      : np.ones(4),
            "coef_free_p7"      : np.ones(4),
            "coef_free_p8"      : np.ones(4),
            "coef_free_p9"      : np.ones(4),

            "power_total"       : np.ones(4),
            "power_free_1"      : np.ones(4),
            "power_free_2"      : np.ones(4),
            "power_free_3"      : np.ones(4),
            "power_free_4"      : np.ones(4),
            "coef_exp_1"        : np.ones(4),
            "coef_exp_2"        : np.ones(4),
            "coef_exp_3"        : np.ones(4),
            "coef_axis_1"       : np.ones(4),
            "coef_axis_2"       : np.ones(4),
            "coef_axis_3"       : np.ones(4)
        }

    def set_value(self, dictsearch_string, v):
        '''
        To set: best value, initial guess, lower bound, upper bound.
        '''
        for i in range(4):
            self.params_dict[dictsearch_string][i] = v[i]



####[] To wrap the target function with its model. The main troublesome process.
class Wrap_func:
    def __init__(self, m, f, params_name=[""], p_fixed=None):
        self.model = m
        self.funcfit_formula = f
        # self.tag_prior = tag0 #prior tag to select
        # self.tag_likelihood = tag1 #likelihood tag to select
        # self.tag_posterior = tag2 #log posterior
        # self.tag_curve = tag3 #curve_fit
        self.set_bounds(params_name)
        self.params_fixed = p_fixed

    def set_bounds(self, params_name=[""]):
        # if not len(params_name)==self.N_fitparams:
        #     print("Wrong length of params! Please check.")
        #     exit(0)

        nf = len(params_name)
        self.N_fitparams = nf
        self.params = list(range(nf))
        self.params_name = list(range(nf))
        self.reference_list = list(range(nf-1)) # donot have "log_penalty"
        self.min_list = list(range(nf-1))
        self.max_list = list(range(nf-1))
        self.fitvalue_list = list(range(nf-1))
        self.p_other = list(range(nf-1))

        self.params_name = params_name
        self.p_other = None
        for i in range(self.N_fitparams-1):
            self.reference_list[i]  = self.model.params_dict[params_name[i]][1]
            self.min_list[i]        = self.model.params_dict[params_name[i]][2]
            self.max_list[i]        = self.model.params_dict[params_name[i]][3]

    def funcfit(self, x, *p): #list of params may not be all of the params of self.funcfit_formula
        if self.params_fixed is not None:
            return self.funcfit_formula(x, *p, *(self.params_fixed))
        else:
            return self.funcfit_formula(x, *p)

    def log_prior(self, theta):
        for i in range(self.N_fitparams): # prior of params, one by one
            if not( self.model.params_dict[self.params_name[i]][2] < theta[i] \
                < self.model.params_dict[self.params_name[i]][3] ):
                # print("Out of prior!")
                return -np.inf
        # return 0.
        if not theta[0]<theta[1]:
            # print("Out of prior!")
            return -np.inf
        else:
            return 0.

    def log_likelihood(self, theta, x, y, yerr):
        abc = theta[0:-1]
        log_f = theta[-1]
        ymodel = self.funcfit(x, *abc)
        ye = yerr
        sigma2 = ye**2 + ymodel**2*np.exp(2*log_f)  # s_n^2
        return -0.5 * np.sum((y-ymodel)**2/sigma2 + np.log(sigma2))  # +C

    def log_posterior(self, theta, x, y, yerr):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta, x, y, yerr)
        return lp + ll

    def assign_fitvalue_list(self, fitvalues): #send fit value to the first value of params_dict of MG model
        for i in range(self.N_fitparams-1):
            self.model.params_dict[self.params_name[i]][0] = fitvalues[i]



####[] fit
# To wrap the leastsq fitting. Not used.
class Minimize_fit:
    def __init__(self):
        pass
    
    def leastsq_residual_log(self, funcfit_log, x0, bounds=None, cons=None, args=None):
        '''
        To wrap the residual function of fitting by minimizing with data.
        '''
        def residual_log(pfit, xdata, ydata_log):
            return funcfit_log(xdata, pfit) - ydata_log
        #[] examples:
        # x0 = (0, 0, 0) #corespond to \param pfit
        # bounds = [[0,None], [0,None], [0,None]] #corespond to \param pfit
        # cons = ({'type': 'ineq', 'fun': lambda x: x[0]**2 - x[1] + x[2]**2},
        #   {'type': 'ineq', 'fun': lambda x: -x[0] - x[1] - x[2]**3 + 20},
        #   {'type': 'eq', 'fun': lambda x: -x[0] - x[1]**2 + 2},
        #   {'type': 'eq', 'fun': lambda x: x[1] + 2 * x[2]**2 - 3})
        # res = spopt.minimize(residual_log, x0=x0, 
        #     args=args, bounds=bounds, constraints=cons
        # )
        res = spopt.leastsq(residual_log, x0=x0, args=args
            # , bounds=bounds, constraints=cons
        ) #The summation has been done in leastsq() instead of in minimize()
        return res

    def minimize_residual_log(self, funcfit_log, x0, bounds=None, cons=None, args=None):
        '''
        To wrap the residual function of fitting by minimizing with data.
        '''
        def residual_log(pfit, xdata, ydata_log):
            return funcfit_log(xdata, pfit) - ydata_log
        #[] examples:
        # x0 = (0, 0, 0) #corespond to \param pfit
        # bounds = [[0,None], [0,None], [0,None]] #corespond to \param pfit
        # cons = ({'type': 'ineq', 'fun': lambda x: x[0]**2 - x[1] + x[2]**2},
        #   {'type': 'ineq', 'fun': lambda x: -x[0] - x[1] - x[2]**3 + 20},
        #   {'type': 'eq', 'fun': lambda x: -x[0] - x[1]**2 + 2},
        #   {'type': 'eq', 'fun': lambda x: x[1] + 2 * x[2]**2 - 3})
        res = spopt.minimize(residual_log, x0=x0, 
            args=args, bounds=bounds, constraints=cons
        )
        # res = spopt.leastsq(residual_log, x0=x0, 
        #     args=args, bounds=bounds, constraints=cons
        # ) #The summation has been done in leastsq() instead of in minimize()
        return res



# To wrap the MCMC fitting. Here one should set params of MCMC method and provide the fit model with fit params.
class MCMC_fit_galaxy:
    def __init__(self, wf, x, y, yerr=None, nw=60, n_step=2000, nb_rate=0.5, nth=1, nd=0, sd=0):
        self.Wrap = wf  # params constraints and fit model
        self.N_fitparams = self.Wrap.N_fitparams  #nf #counts of params
        self.x = x  # argumanet x
        self.y = y  # argumanet y
        self.yerr = None  # argumanet y error
        if yerr is None:
            self.yerr = np.zeros(np.shape(self.y))
        else:
            self.yerr = yerr
        self.N_walkers = nw
        self.N_steps = n_step
        self.N_burn = self.N_steps*nb_rate
        self.N_thin = nth
        self.N_discard = nd
        self.seed = sd
        self.result = 0.
        self.sampler = 0.
        # np.ones(self.N_fitparams)
        self.labels = list(range(self.N_fitparams))
        self.true = list(range(self.N_fitparams))
        self.params_MC = list(range(self.N_fitparams))
        self.error_fit = list(range(self.N_fitparams))

    def run(self):
        np.random.seed(self.seed)
        starting_guesses = np.random.random(
            (self.N_walkers, self.N_fitparams))*1.+0.5
        for i in range(self.N_fitparams-1):  # 0:-1
            self.true[i] = self.Wrap.reference_list[i]
            starting_guesses[:, i] *= self.true[i]
        self.true[-1] = -10.  # -1
        starting_guesses[:, -1] *= self.true[-1]

        ncpu = cpu_count()
        print("{0} CPUs".format(ncpu))
        # count_paral = int(ncpu/2)
        count_paral = 1
        print("count of parallels: ", count_paral)
        with Pool(processes=count_paral) as pool:
        # with MPIPool() as pool:
        #     if not pool.is_master():
        #         pool.wait()
        #         sys.exit(0)

            sampler = emcee.EnsembleSampler(self.N_walkers, self.N_fitparams, self.Wrap.log_posterior,
                                            args=[self.x, self.y, self.yerr], pool=pool)
            # () discard
            starttime = time.time()
            sampler.run_mcmc(starting_guesses, self.N_steps, progress=True)
            # sampler.chain
            # self.result = sampler.chain[:, self.N_burn:, :]
            self.sampler = sampler
            endtime = time.time()
            multi_time = endtime - starttime
            print("Multiprocessing took {0:.1f} seconds.".format(multi_time))

    def display(self, name="", dpi=300, is_show=0, current_time="current_time/"):
        samples = self.sampler.get_chain()
        fig, axes = plt.subplots(
            self.N_fitparams, figsize=(10, 7), sharex=True)
        for i in range(self.N_fitparams):
            self.labels[i] = "param %d" % (i)
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(self.labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        fig_tmp = plt.gcf()
        # mkdir current_time/ #if not exsit #for all savefig
        
        fig_tmp.savefig("savefig/"+name+"_MCMC_steps.png", format='png', dpi=dpi) #'eps' 'pdf' 'png'
        if is_show == 1:
            plt.show()
        # tau = self.sampler.get_autocorr_time()
        flat_samples = self.sampler.get_chain(
            flat=True, thin=self.N_thin, discard=self.N_discard
        )
        fig = corner.corner(flat_samples, labels=self.labels, truths=self.true)
        fig_tmp = plt.gcf()
        fig_tmp.savefig("savefig/"+name+"_MCMC_corner.png", format='png', dpi=dpi)
        if is_show == 1:
            plt.show()
        print("Fig ... Done.")
        plt.close("all")
        print("samples and flat_samples shapes: ",
              samples.shape, flat_samples.shape)
        p = self.params_MC
        pf = self.params_MC
        for i in range(self.N_fitparams):
            p[i] = np.percentile(samples[:, i], [16, 50, 84])  # np.mean
            pf[i] = np.percentile(flat_samples[:, i], [16, 50, 84])
            self.params_MC[i] = pf[i][1]
        # add.DEBUG_PRINT_V(0, p, pf, self.params_MC)
        ymodel = self.Wrap.funcfit(self.x, *self.params_MC[0:-1])
        self.error_fit = [add.norm_l(ymodel, self.y, axis=0) / len(self.y)]
        print("MCMC fit result: ", self.params_MC, self.error_fit)
        
        self.optimization = self.params_MC[:-1]
        self.covariance = None
        self.ymodel = self.Wrap.funcfit(self.x, *self.optimization)
        self.error_fit = [add.norm_l(self.ymodel, self.y, axis=0) / len(self.y)]
        self.residual = self.ymodel-self.y
        self.sigma = np.std(self.residual)
        self.residual_sigma = self.residual/self.sigma
        print("MCMC result:: ")
        print("reference value list: ", self.Wrap.reference_list)
        print("optimization params list: ", self.optimization)
        print("error_fit: ", self.error_fit)
        return self.params_MC, self.covariance, self.residual_sigma, self.error_fit, self.ymodel



# To wrap the curve_fit fitting.
class Curve_fit_galaxy:
    def __init__(self, wf, x, y, yerr=None, tag0=0, mf=5000):
        self.Wrap = wf
        # self.N_fitparams = nf
        self.maxfev = mf
        self.x = x
        self.y = y
        self.yerr = None  # argumanet y error
        if yerr is None:
            self.yerr = None #np.zeros(np.shape(self.y))
        else:
            self.yerr = yerr
        self.tag_fit = tag0
        self.optimization = 0.
        self.covariance = 0.
        self.error_fit = 0.
        self.residual = 0.
        self.residual_sigma = 0.

    def run(self):
        print("curve_fit settings:: ")
        print("reference value list: ", self.Wrap.reference_list)
        # print("bounds of min and max list: ", self.Wrap.min_list, self.Wrap.max_list)
        print("curve_fit running...")
        self.optimization, self.covariance = curve_fit(self.Wrap.funcfit, self.x, self.y, sigma=self.yerr, 
            p0=self.Wrap.reference_list, bounds=(self.Wrap.min_list, self.Wrap.max_list), maxfev=self.maxfev)
        # plsq = spopt.leastsq(self.Wrap.funcfit_leastsq, self.Wrap.reference_list, args=(self.y, self.x))
        print("curve_fit running... Done.")
        self.Wrap.assign_fitvalue_list(self.optimization)
        print("Assign fit values to model_galaxy ... Done.")

    def minimize_res(self):
        # log_residuals_dpl(p0, x, y)
        # res = optimize.leastsq(log_residuals_dpl,x0=guess,args=(dens,xyz))
        print("minimize_res settings:: ")
        print("reference value list: ", self.Wrap.reference_list)
        # print("bounds of min and max list: ", self.Wrap.min_list, self.Wrap.max_list)
        print("curve_fit running...")
        # self.optimization, self.covariance = curve_fit(self.Wrap.funcfit, self.x, self.y, sigma=self.yerr,
        #     p0=self.Wrap.reference_list, bounds=(self.Wrap.min_list, self.Wrap.max_list), maxfev=self.maxfev)
        # plsq = spopt.leastsq(self.Wrap.funcfit_leastsq, self.Wrap.reference_list, args=(self.y, self.x))
        # pmnz = spopt.minimize(self.Wrap.funcfit_leastsq, self.Wrap.reference_list, args=(self.y, self.x))
        plsq = spopt.leastsq(self.Wrap.funcfit_leastsq, self.Wrap.reference_list, args=(self.y, self.x))
        print("curve_fit running... Done.")
        self.Wrap.assign_fitvalue_list(self.optimization)
        print("Assign fit values to model_galaxy ... Done.")
        return 

    def display(self):
        self.ymodel = self.Wrap.funcfit(self.x, *self.optimization)
        self.error_fit = [add.norm_l(self.ymodel, self.y, axis=0) / len(self.y)]
        self.residual = self.ymodel-self.y
        self.sigma = np.std(self.residual)
        self.residual_sigma = self.residual/self.sigma
        print("curve fit result:: ")
        print("reference value list: ", self.Wrap.reference_list)
        print("optimization params list: ", self.optimization)
        # print("covariance: ", self.covariance)
        # print("residual: ", self.residual)
        print("residual_sigma: ", self.residual_sigma)
        print("error_fit: ", self.error_fit)
        return self.optimization, self.covariance, self.residual_sigma, self.error_fit, self.ymodel



####[] plot fit
class Plot_model_fit:
    def __init__(self):
        # plot options
        self.is_plotlog = 0
        self.color = ["purple", "blue", "green",
                      "orange", "red"]  # color value
        self.figsize = (20, 16)
        self.dpi = 100
        self.pointsize = 2.
        self.fontsize = 30.
        self.axis_scale = "log"
        self.title = ""
        # declaration
        self.x = 0.
        self.y = 0.
        self.xerr = 0.
        self.yerr = 0.
        self.funcfit = 0.
        self.params = 0.
        self.standard = 0.
        self.ll = 0.
        self.datalist = 0.
        self.fig_all = 0.

    def load(self, *datalist):
        self.ll = len(datalist)
        self.datalist = datalist

    def set_figure(self, **args):
        pass

    def plot(self, name="", xl="", yl="", text="", is_relative=0, id_relative_compare=1, dpi=300, is_show=0):
        self.xlabel = xl
        self.ylabel = yl
        self.fig_all = plt.figure(figsize=self.figsize, dpi=self.dpi)
        l_index = 0
        compare_funcfit = self.datalist[id_relative_compare][6]
        compare_x = self.datalist[id_relative_compare][0]
        compare_params = self.datalist[id_relative_compare][7]
        compare = compare_funcfit(compare_x, *compare_params)
        for l in self.datalist:
            # load each group of data
            x = l[0]
            y = l[1]
            xerr = l[2]
            yerr = l[3]
            xp = l[4]
            yp = l[5]
            funcfit = l[6]
            params = l[7]
            others = l[8]
            label = l[9]
            # plot each
            # plt.subplot(self.ll,l)
            ym = funcfit(x, *params)
            ym_plot = ym
            y_plot = y
            self.standard = [add.norm_l(ym, y, axis=0) / len(y)]
            # self.standard = others
            # if l_index == id_relative_compare:
            #     compare = ym
            if is_relative == 1:
                ym_plot = ym/compare-1.
                y_plot = y/compare-1.
            if l_index == 0:
                plt.scatter(xp, y_plot, s=self.pointsize,
                            color="k", label="data generated by DICE\n"+text)
            plt.scatter(xp, ym_plot, s=self.pointsize,
                        label=label+", ye=%e" % (self.standard[0]))
            l_index += 1
        # plt.xscale(self.axis_scale)
        # plt.yscale(self.axis_scale)
        plt.rcParams["legend.markerscale"] = self.pointsize*10.
        plt.tick_params(labelsize=self.fontsize)
        plt.legend(fontsize=self.fontsize)
        plt.xlabel(self.xlabel, fontsize=self.fontsize)
        plt.ylabel(self.ylabel, fontsize=self.fontsize)
        if is_relative != 1:
            plt.xlim(0., 250.) #log
            plt.ylim(-30., 0.)
        plt.title(self.title, fontsize=self.fontsize)
        fig_tmp = plt.gcf()
        fig_tmp.savefig("savefig/funcfit/"+name+"_compare_isrelative%d.png" %
                        (is_relative), format='png', dpi=dpi, bbox_inches='tight')
        if is_show == 1:
            plt.show()
        print("Fig ... Done.")
        plt.close("all")

    def plot3d(self):
        pass

    def plot_actions_Comb_NDF(self, ddl, nm="", text="", lim=[False], is_show=1):

        pointsize = 0.06
        fontsize = 6.0
        dpi = 500
        fig = plt.figure(dpi=300)
        # ax=fig.add_subplot(111, projection='3d') # ax = Axes3D(fig)
        ax=fig.add_subplot(111)
        ax.grid(True) # ax.set_axis_off()

        for i in range(len(ddl)):
            x = ddl[i][0]
            y = ddl[i][1]
            lb = ddl[i][-1]
            ax.scatter(x, y, s=pointsize, label=lb, marker="+")
        ax.legend(fontsize=fontsize, loc=0)

        ax.set_xlabel(r"x", fontsize=fontsize)
        ax.set_ylabel(r"y", fontsize=fontsize)
        ax.set_title(r"actions_Comb_NDF_"+"%s"%(text), fontsize=fontsize)
        # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)
        
        if(lim[0]):
            ax.set_xlim(lim[1], lim[2])
            ax.set_ylim(lim[3], lim[4])

        fig_tmp = plt.gcf()
        fig_tmp.savefig("savefig/actions_Comb_NDF_%s.png"%(nm), format='png', dpi=dpi, bbox_inches='tight')
        if is_show == 1:
            plt.show()
        print("Fig ... Done.")
        plt.close("all")
        return 

    def plot_actions_Comb_NDF_subplot(self, ddll, subtitle="", xl=[], yl=[], lim=[False], is_show=1):

        L = len(ddll)
        pointsize = 0.6
        fontsize = 12.0
        # dpi = dpi
        # fig = plt.figure()
        fig = plt.figure(figsize=(16,10), dpi=200)

        for llll in range(L):
            ddl = ddll[llll]
            ax = fig.add_subplot(int(np.ceil(L/2.)), 2, llll+1)
            ax.grid(True) # ax.set_axis_off()

            for i in range(len(ddl)):
                x = ddl[i][0]
                y = ddl[i][1]
                lb = ddl[i][-1]
                ax.scatter(x, y, s=pointsize, label=lb, marker="+")
            ax.legend(fontsize=fontsize/2)#, loc=0)

            ax.set_xlabel(xl[llll], fontsize=fontsize)
            ax.set_ylabel(yl[llll], fontsize=fontsize)
            # ax.set_title(r"actions_Comb_NDF_"+"%s"%(text), fontsize=fontsize)
            # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)
            
            if lim[0]:
                # xlim = (np.min(x), np.max(x))
                xlim = (0., np.median(x)*8)
                if llll==1:
                    ylim = (-0.8, 0.8)
                else:
                    # ylim = (np.min(y), np.max(y))
                    # ylim = (np.median(y)*4, np.median(y)*4)
                    ylim = (-18., -8.)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)


        # whspace = 0.4
        # plt.subplots_adjust(hspace=whspace, wspace=whspace)
        plt.suptitle(subtitle)

        # w = RGB.shape[0]
        # h = RGB.shape[1]
        # dpi = 300
        # fig = plt.figure(figsize=(w/dpi,h/dpi),dpi=dpi)
        # axes = fig.add_axes([0,0,1,1])
        # axes.set_axis_off()
        # axes.imshow(RGB)
        # manager = plt.get_current_fig_manager()
        # manager.window.showMaximized()
        # # manager.resize(*manager.window.maxsize()
        # # manager.frame.Maximize(True)
        # plt.gcf().set_size_inches(512 / 100, 512 / 100)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # # plt.subplots_adjust(top=1, bottom=0, right=0.93, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.savefig("savefig/%s.png"%(subtitle), format='png', bbox_inches='tight')
        
        # # plt.clf()
        # plt.tight_layout()
        # plt.savefig("savefig/%s.png"%(subtitle), format='png', bbox_inches='tight')
        # plt.close()
        # plt.show()

        fig_tmp = plt.gcf()
        fig_tmp.savefig("savefig/%s.png"%(subtitle), format='png', bbox_inches='tight')
        # if is_show == 1:
        #     plt.show()
        print("Fig ... Done.")
        plt.close("all")
        return 

    def plot_x_scatter3d_dd(self, ddl, nm="", text="", is_lim=False, bd=None, k_median=0., is_show=1):

        fig = plt.figure(dpi=300)
        pointsize = 0.2
        fontsize = 6.0
        dpi = 500
        ax=fig.add_subplot(111,projection='3d') # ax = Axes3D(fig)
        ax.grid(True) # ax.set_axis_off()

        for i in range(len(ddl)):
            x = ddl[i][0]
            f = ddl[i][1]
            lb = ddl[i][-1]
            # cm = plt.cm.get_cmap('RdYlBu') #red-yellow-blue
            cm = plt.cm.get_cmap('gist_rainbow') #rainbow
            axsc = ax.scatter(x[:,0], x[:,1], x[:,2], s=pointsize, label=lb, c=f, cmap=cm)
            plt.colorbar(axsc)
        # surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
        # position=fig.add_axes([0.1, 0.3, 0.02, 0.5]) #x pos, ypos, width, shrink
        # fig.colorbar(surf, cax=position, aspect=5)
        # # ax.arrow(-lim,0, 2*lim,0) #only 2d??
        # ax.plot3D([-lim,lim],[0,0],[0,0], color="red", linewidth=pointsize)
        # ax.plot3D([0,0],[-lim,lim],[0,0], color="red", linewidth=pointsize)
        # ax.plot3D([0,0],[0,0],[-lim,lim], color="red", linewidth=pointsize)
        ax.legend(fontsize=fontsize, loc=0)

        ax.set_xlabel(r"x", fontsize=fontsize)
        ax.set_ylabel(r"y", fontsize=fontsize)
        ax.set_zlabel(r"z", fontsize=fontsize)
        ax.set_title(r"scatter3d_dd_"+"%s"%(text), fontsize=fontsize)
        # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)
        
        if is_lim:
            if bd!=None:
                ax.set_xlim(0., bd[0])
                ax.set_ylim(0., bd[1])
                ax.set_zlim(0., bd[2])
            else:
                # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.6, 1., 0.6, 1.]))
                # klim = 0.9
                # lim=200.
                # ax.set_xlim(-lim, lim)
                # ax.set_ylim(-lim, lim)
                # ax.set_zlim(-lim, lim)
                klim = k_median
                lim = np.median(x,axis=0)*klim
                minx = np.min(x,axis=0)
                # ax.set_xlim(-lim[0]*0., lim[0])
                # ax.set_ylim(-lim[1]*0., lim[1])
                # ax.set_zlim(-lim[2]*0., lim[2])
                ax.set_xlim(minx[0], lim[0])
                ax.set_ylim(minx[1], lim[1])
                # ax.set_zlim(minx[2], lim[2])

        # ax.set_xscale("log")
        # ax.set_yscale("log")
        # ax.set_zscale("log")

        fig_tmp = plt.gcf()
        fig_tmp.savefig("savefig/scatter3d_dd_action_%s.png"%(nm), format='png', dpi=dpi, bbox_inches='tight')
        if is_show == 1:
            plt.show()
        print("Fig ... Done.")
        plt.close("all")

    def plot_histogram(self, x_list, bins_list=None, label_list=None, 
        name="", is_log=False, limit_list=None, xyzlabel_list=None, is_save=False
    ):
        pointsize = 0.2
        fontsize = 6.0
        dpi = 400
        fig = plt.figure()
        for j in range(len(x_list)):
            ax = fig.add_subplot(len(x_list),1,j+1)
            for i in range(len(x_list[j])):
                x = x_list[j][i]
                xbins = np.linspace(-np.min(x), np.max(x), 10)
                if bins_list!=None:
                    xbins = bins_list[j][i]
                lb = None
                if label_list!=None:
                    lb = label_list[j][i]
                h, xedge = np.histogram(x, bins=xbins)
                ax.step(xbins[:-1], h, label=lb, where='post')

            text = r"scatter3d_"+"%s"%(name)
            ax.legend(fontsize=fontsize, loc=0)
            ax.set_title(text, fontsize=fontsize)
            # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)

        fig_tmp = plt.gcf()
        if is_save:
            fig_tmp.savefig("savefig/x_scatter/"+text+".png", format='png', dpi=dpi, bbox_inches='tight')
        plt.show()
        plt.close("all")
        return 

    def plot_x_scatter3d_general(self, x_list, f_list=None, label_list=None, 
        name="", is_log=False, limit_list=None, xyzlabel_list=None, is_save=False
    ):
        pointsize = 0.2
        fontsize = 6.0
        dpi = 400
        fig = plt.figure(figsize=None, dpi=dpi)
        ax=fig.add_subplot(1,1,1, projection='3d')
        ax.grid(True) #ax.set_axis_off()

        for i in range(len(x_list)):
            x = x_list[i]
            lb = None
            if label_list!=None:
                lb = label_list[i]
            if f_list!=None:
                f = f_list[i]
                # f = f_list[-1]
                cm = plt.cm.get_cmap('gist_rainbow') #rainbow
                axsc = ax.scatter(x[:,0], x[:,1], x[:,2], s=pointsize, label=lb, c=f, cmap=cm)
                plt.colorbar(axsc)
            else:
                axsc = ax.scatter(x[:,0], x[:,1], x[:,2], s=pointsize, label=lb)
            
        if limit_list!=None:
            ax.set_xlim(limit_list[0], limit_list[1])
            ax.set_ylim(limit_list[2], limit_list[3])
            ax.set_zlim(limit_list[4], limit_list[5])
            ax.plot([limit_list[0], limit_list[1]], [0,0], [0,0], lw=pointsize*2, color="black")
            ax.plot([0,0], [limit_list[2], limit_list[3]], [0,0], lw=pointsize*2, color="black")
            ax.plot([0,0], [0,0], [limit_list[4], limit_list[5]], lw=pointsize*2, color="black")

        if xyzlabel_list!=None:
            ax.set_xlabel(r"%s"%(xyzlabel_list[0]), fontsize=fontsize)
            ax.set_ylabel(r"%s"%(xyzlabel_list[1]), fontsize=fontsize)
            ax.set_zlabel(r"%s"%(xyzlabel_list[2]), fontsize=fontsize)
        else:
            ax.set_xlabel(r"coordinate1", fontsize=fontsize)
            ax.set_ylabel(r"coordinate2", fontsize=fontsize)
            ax.set_zlabel(r"coordinate3", fontsize=fontsize)

        if is_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_zscale("log")

        text = r"scatter3d_"+"%s"%(name)
        ax.legend(fontsize=fontsize, loc=0)
        ax.set_title(text, fontsize=fontsize)
        # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)

        fig_tmp = plt.gcf()
        if is_save:
            fig_tmp.savefig("savefig/x_scatter/"+text+".png", format='png', dpi=dpi, bbox_inches='tight')
        plt.show()
        plt.close("all")

    def plot_x_scatter3d_xxx(self, ddl, nm="", text="", is_lim=False, k_median=8., is_show=1):

        fig = plt.figure(dpi=300)
        pointsize = 0.2
        fontsize = 6.0
        dpi = 500
        ax=fig.add_subplot(111,projection='3d') # ax = Axes3D(fig)
        ax.grid(True) # ax.set_axis_off()

        for i in range(len(ddl)):
            x = ddl[i][0]
            lb = ddl[i][-1]
            # cm = plt.cm.get_cmap('gist_rainbow') #rainbow
            # ax.scatter(x[:,0], x[:,1], x[:,2], s=pointsize, label=lb, c=f, cmap=cm)
            # plt.colorbar(axsc)
            ax.scatter(x[:,0], x[:,1], x[:,2], s=pointsize, label=lb)
        ax.legend(fontsize=fontsize, loc=0)

        name = r"scatter3d_xxx_"+"%s"%(text)
        ax.set_xlabel(r"x", fontsize=fontsize)
        ax.set_ylabel(r"y", fontsize=fontsize)
        ax.set_zlabel(r"z", fontsize=fontsize)
        ax.set_title(name, fontsize=fontsize)
        # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)
        
        if is_lim:
            klim = k_median
            lim = np.median(x,axis=0)*klim
            minx = np.min(x,axis=0)
            # ax.set_xlim(-lim[0]*0., lim[0])
            # ax.set_ylim(-lim[1]*0., lim[1])
            # ax.set_zlim(-lim[2]*0., lim[2])
            ax.set_xlim(minx[0], lim[0])
            ax.set_ylim(minx[1], lim[1])
            # ax.set_zlim(minx[2], lim[2])

        # ax.set_xscale("log")
        # ax.set_yscale("log")
        # ax.set_zscale("log")

        fig_tmp = plt.gcf()
        fig_tmp.savefig("savefig/scatter3d_xxx_action_%s.png"%(nm), format='png', dpi=dpi, bbox_inches='tight')
        if is_show == 1:
            plt.show()
        print("Fig ... Done.")
        plt.close("all")



class Wrapper_for_fit_function:
    def __init__(self, func, argumentname_all):
        self.func = func
        self.arguments_name = argumentname_all #list
        self.arguments_boundary = None #list
        self.arguments_reference = None #list
        self.arguments_optimized = None #list
        self.arguments_count = len(argumentname_all) #int
        self.arguments = list(range(self.arguments_count)) #dict
        #?? dict

    def set_argument_someone(self, argname, values, argnum=None):
        # if argnum is not None and type(argnum)==int:
        self.arguments[argname] = values

    def set_boundary_onebyone(self, boundary=None):
        self.arguments_boundary = boundary
        return 0

    def set_reference_onebyone(self, reference=None):
        self.arguments_reference = reference
        return 0

    def other(self):
        return 0



####mpfit
import mpfit as mf
class MP_fit:
    '''
    A wrapper to mpfit, not complete.

    # example code:
    def Func(x, p):
        return p[0] + p[1]*x + p[2]*x**2 + p[3]*np.sqrt(x) +p[4]*np.log(x)
    xd = np.arange(100)+1.
    p_fact = [5.7, 2.2, 500., 1.5, 20.]
    yd = p_fact[0] + p_fact[1]*xd + p_fact[2]*xd**2 + p_fact[3]*np.sqrt(xd) + p_fact[4]*np.log(xd)
    errd = None
    p0 = [15.7, 0.2, 1500., 0.5, 200.]
    functkw = {"x": xd, "y": yd, "err": errd} #mpfit
    MF = fgw.MP_fit(Func, p0, functkw)
    mparams = MF.run()
    print(mparams)
    # code end
    '''
    def __init__(self, Func, p0, bounds=None, fixed=None, 
                 functkw=None, parinfo=None, 
                 fjac=None, xdata=None, ydata=None, errdata=None, 
                 maxiter=200, quiet=0):
        self.Func = Func
        self.p0 = p0
        self.functkw = None
        if functkw is None:
            self.functkw = {"x": xdata, "y": ydata, "err": errdata} #mpfit
        else:
            self.functkw = functkw
        self.parinfo = None
        if parinfo is None:
            np0 = len(p0)
            self.parinfo = [{
                    'value':p0[i], 
                    # 'fixed':0, 
                    'limited':[1, 1], 
                    'limits':[bounds[0][i], bounds[1][i]]
                } for i in range(np0)]
        else:
            self.parinfo = parinfo
        self.maxiter = maxiter
        self.quiet = quiet

    def myfunct(self, p, fjac=None, x=None, y=None, err=None):
        '''
        The mpfit has not x_error. One may use the propagation formula of error to fit data with x_error.
        '''
        # Parameter values are passed in "p"
        # If fjac==None then partial derivatives should not be
        # computed.  It will always be None if MPFIT is called with default
        # flag.
        # model = self.Func(x, p)
        model = self.Func(x, *p)
        # Non-negative status value means MPFIT should continue, negative means
        # stop the calculation.
        status = 0
        if err is None: #In a typical scientific problem the residuals should be weighted 
            # so that each deviate has a gaussian sigma of 1.0
            # return [status, (yd-model)/yd] #?? err = yd-yd_measure #perror in mpfit
            return [status, (y-model)/1.] #?? err = yd-yd_measure #perror in mpfit
        if err is not None:
            return [status, (y-model)/err]

    def myfunct_log(self, p, fjac=None, x=None, y=None, err=None): #?? +- of y_error
        '''
        Similar to function self.myfunct, but one fit log(y) with log_y_data and log_y_error.
        Setting 
            res_0 = |y_model-y_data|/|y_error|,
        if |res_0-1|<1, i.e., 0<res_0<2 for real value,
        then after Tylor epanding to 2-order,
        we have 
            res_log = log(|y_model-ydata|)/log(|yerror|) 
                    = 1-(res_0^2-4*res_0+3)/(2*log(y_error))+O(res_0^3).
        '''
        model = self.Func(x, *p)
        status = 0
        if err is None:
            return [status, (y-model)/1.]
        if err is not None:
            res_0 = np.abs(y-model)/np.abs(err)
            res_log = 1.-(res_0**2-4.*res_0+3.)/(2*err)
            return [status, res_log]

    def run(self, is_fit_log=False):
        print("Start to run mpfit.")
        if is_fit_log:
            m = mf.mpfit(self.myfunct_log, self.p0, functkw=self.functkw, parinfo=self.parinfo, maxiter=self.maxiter, quiet=self.quiet)
        else:
            m = mf.mpfit(self.myfunct, self.p0, functkw=self.functkw, parinfo=self.parinfo, maxiter=self.maxiter, quiet=self.quiet)
        print("status = ", m.status)
        if (m.status <= 0):
            print("error message = ", m.errmsg)
        print("End to run mpfit, m.params = ", m.params)
        return m.params



####[] main()
if __name__ == '__main__':
    
    # ##debug
    # p = 1,1,1,1
    # q = Wrap_func(1, *p)

    '''
    ## usual settings
    M = 137.
    N_ptcs = 10000
    ls = 19.6
    ds = 0.004568 #0.000891
    Js = (gm.G*M*ls)**0.5
    ar_expect = [1., 0.6, 0.3]
    id_relative_compare = 1

    MG = Model_galaxy(M, ls, ds)
    # coef_boundary = 1.e2
    # coef_boundary = 1.e8
    coef_boundary = 1.e40
    # scale_boundary = 1.e1
    scale_boundary = 1.e2
    # power_boundary = 1.e1
    power_boundary = 1.e3
    axisratio_boundary = 1.e2
    MG.set_value("density_scale",   np.array([ds, ds, ds/scale_boundary, ds*scale_boundary]))
    MG.set_value("length_scale",    np.array([ls, ls, ls/scale_boundary, ls*scale_boundary]))
    MG.set_value("rotate_angle_x",  np.array([0., 2*np.pi, 0., 2*np.pi])) # np.pi #divide by zero
    MG.set_value("rotate_angle_y",  np.array([0., 2*np.pi, 0., 2*np.pi]))
    MG.set_value("rotate_angle_z",  np.array([0., 2*np.pi, 0., 2*np.pi]))
    MG.set_value("action_scale",    np.array([Js, Js, Js*1e-1, Js*1e1]))
    MG.set_value("log_penalty",     np.array([-10., -10., -100., 1.]))
    # MG.set_value("coef_total",      np.array([1., 1., 1.-0.1, 1.+0.1]))
    # MG.set_value("coef_total",      np.array([1., 1., 1.e-3, 1.e3]))
    MG.set_value("coef_total",      np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_n3",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_n2",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_n1",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_0",     np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p1",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p2",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p3",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p4",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_exp_1",      np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_exp_2",      np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_exp_3",      np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_axis_1",     np.array([1., 1., 1./axisratio_boundary, axisratio_boundary]))
    MG.set_value("coef_axis_2",     np.array([1., 1., 1./axisratio_boundary, axisratio_boundary]))
    MG.set_value("coef_axis_3",     np.array([1., 1., 1./axisratio_boundary, axisratio_boundary]))
    # MG.set_value("power_alpha",     np.array([1., 1., 1e-4,             1e1]))
    # MG.set_value("power_beta",      np.array([3., 3., 2.e-1,            1e2]))
    MG.set_value("power_alpha",     np.array([1., 1., 0.,               power_boundary]))
    MG.set_value("power_beta",      np.array([3., 3., 0.,               power_boundary]))
    MG.set_value("power_total",     np.array([1., 1., 1./power_boundary,               power_boundary]))
    MG.set_value("power_free_1",    np.array([1., 1., 1./power_boundary,               power_boundary]))
    MG.set_value("power_free_2",    np.array([1., 1., 1./power_boundary,               power_boundary]))
    MG.set_value("power_free_3",    np.array([1., 1., 1./power_boundary,               power_boundary]))
    MG.set_value("power_free_4",    np.array([1., 1., 1./power_boundary,               power_boundary]))
    MG.set_value("power_Einasto",   np.array([1., 1., 1./power_boundary,               power_boundary]))

    gm_name = ""
    # gm_name = "_1_NFW_spherical"
    # gm_name = "_4_EinastoUsual_spherical"
    # gm_name = "_11_NFW_triaxial"
    # gm_name = "_41_EinastoUsual_triaxial"
    galaxymodel_name = "galaxy_general"+gm_name+"/"
    snapshot_Id = 5000

    whatcannonical = 5
    # method_and_tags = "C5P0S2A0" #SFFP
    # method_and_tags = "C5P0S2A2" #GFFP
    # method_and_tags = "C5P1S2A0" #SFDP
    method_and_tags = "C5P1S2A1" #TEPPOD

    # bd = 1e5
    # bd = 1e5/2
    bd = 1e6
    bd_display = bd
    N_neighbour = 64

    ## each
    galaxymodel_name = "galaxy_general"+gm_name+"/"
    RD = Read_data_galaxy(MG, gmn=galaxymodel_name, wc=whatcannonical) #No such type of data provided
    data = RD.data_original_NDFA(snapshot_Id, method_and_tags=method_and_tags)
    x, y = RD.data_sample_screen(data, x_down=1./bd, x_up=bd, is_logy=True, is_abs=True)
    xerr = x*0.
    yerr = y*1.
    add.DEBUG_PRINT_V(1, x.shape, y.shape)
    # x = np.hstack((x[:,3:6], x[:,0:3])) #now J[3], O[3]
    x = np.hstack((x[:,3:6], np.ones(x[:,0:3].shape))) #now J[3], O[3] #TEPPOD
    Jes = gm.AA_combination_estimateScale(x)
    print("AA scale from data: %e" % Jes)
    # add.DEBUG_PRINT_V(0, np.mean(x,axis=1), np.mean(x,axis=0), np.median(x,axis=0))
    # add.DEBUG_PRINT_V(0, "here1")



    ## fit gal from density??
    # 
    # and axisRatio
    ar = [1., 1., 1.]
    '''



    '''
    ## comb 1 and DF
    fitmodel = [
        gm.AA_fCombination_exp, gm.AA_fCombination_exppower_log, gm.AA_fCombination_P1Rpower_log, 
        gm.AA_fCombination_power1_log, gm.AA_fCombination_polysum_log, #gm.AA_fCombination_Posti15_log, 
        gm.AA_fCombination_powermultWithoutMC_log, gm.AA_fCombination_powermult_log
        # dpl, multipowerlaw
    ]
    fitmodel_name = [
        "gm.AA_fCombination_exp", "gm.AA_fCombination_exppower_log", "gm.AA_fCombination_P1Rpower_log", 
        "gm.AA_fCombination_power1_log", "gm.AA_fCombination_polysum_log", #"gm.AA_fCombination_Posti15_log", 
        "gm.AA_fCombination_powermultWithoutMC_log", "gm.AA_fCombination_powermult_log"
    ]
    N_fitmodel = len(fitmodel)
    fitmodel_params = list(range(N_fitmodel))
    fitmodel_params_name = [
        ["coef_total", "log_penalty"], 
        ["coef_total", "power_Einasto", "log_penalty"], 
        ["coef_total", "power_total", "log_penalty"],  
        ["coef_total", "power_free_1", "log_penalty"], 
        ["coef_free_n3", "coef_free_n2", "coef_free_n1", "coef_free_0", "coef_free_p1", 
            "coef_free_p2", "coef_free_p3", "log_penalty"], 
        # ["power_free_1", "power_free_2", "log_penalty"],
        ["power_free_1", "power_free_2", "power_free_3", "power_free_4", "log_penalty"], 
        ["coef_total", "power_free_1", "power_free_2", "power_free_3", "power_free_4", "log_penalty"]
    ]

    combmodel = [gm.AA_combination_sum, gm.AA_combination_sumWeightFrequency_rateF1, gm.AA_combination_norml2, 
        gm.AA_combination_AxisRatio, 
        gm.AA_combination_norml2WeightFrequency, 
        gm.AA_combination_free_sum, gm.AA_combination_free_AxisRatio, gm.AA_combination_freefree_sumPower]
    combmodel_name = ["gm.AA_combination_sum", "gm.AA_combination_sumWeightFrequency_rateF1", "gm.AA_combination_norml2",
        "gm.AA_combination_AxisRatio", 
        "gm.AA_combination_norml2WeightFrequency", 
        "gm.AA_combination_free_AxisRatio", "gm.AA_combination_free_sum", "gm.AA_combination_freefree_sumPower"]
    N_combmodel = 4 #len(combmodel)
    combmodel_params = [[], [], [], ar, [], [0., 0., 0.], [0., 0., 0.], [0., 0., 0., 0., 0., 0.]]
    
    DD_name = ["x:Comb(J3)*1eX, y:NDF", "x:Comb(J3)*1eX, y:funcFit(J3)", "x:Comb(J3)*1eX, y:errorFit", 
        "x:Comb(J3)*1eX, y:NDF; neighbourAverage", "x:Comb(J3)*1eX, y:NDF; neighbourScatter"]
    N_DD = len(DD_name)
    description = list(range(N_DD))

    AA_data = x #data
    A_data = x[:, 0:3] #data
    DF_data = y #data
    CB = list(range(N_combmodel)) #data
    DF = list(range(N_fitmodel)) #data
    for k in range(len(DF)):
        DF[k] = []
    PM = list(range(N_fitmodel)) #params
    for k in range(len(PM)):
        PM[k] = []
    DD = [[[0 for k in range(N_DD)] for i in range(N_combmodel)] for j in range(N_fitmodel)] #decrease dimension and plot

    #fitmodel
    for j in np.arange(N_fitmodel):
        #usual comb: -xcf -edms
        for i in np.arange(N_combmodel):
            print(j,i)
            # x
            # y
            CB[i] = combmodel[i](AA_data, *(combmodel_params[i]))
            # add.DEBUG_PRINT_V(1, j, i, CB[i].shape, DF_data.shape)

            nf = len(fitmodel_params_name[j])
            WF = Wrap_func(MG, fitmodel[j], nf)
            WF.set_bounds(fitmodel_params_name[j])
            ff = WF.funcfit
            # nf = WF.N_fitparams
            pe = np.array(WF.reference_list)
            CF = Curve_fit_galaxy(WF, nf, CB[i], DF_data, yerr, tag0=0, mf=5000)
            CF.run()
            # add.DEBUG_PRINT_V(1, "AA_fCombination_exp_E")
            P_CF, covariance, residual, residual_sigma, E_CF = CF.display()
            DF_CF = ff(CB[i], *P_CF)
            DF[j].append(DF_CF)
            PM[j].append(P_CF)

            Js = np.median(CB[i])
            # add.DEBUG_PRINT_V(0, Js, min(CB[i]), max(CB[i]), "Js")
            Js0 = np.median(np.sum(x[:,0:3], axis=1)) #for log DF??
            CB_Js = CB[i]/Js
            cf_i = add.merge_array_by_hstack([CB[i], DF_data])
            cf_lm_i, cf_ls_i, cf_f_sortIndex_i = add.neighbourAverage_bin(cf_i, which_column=-1, N_neighbour=N_neighbour, is_all=1)
            CB_lm_i = cf_lm_i[:,0]
            CB_ls_i = cf_ls_i[:,0]#/cf_lm_i[:,0]
            DF_i = cf_lm_i[:,1]

            # ddx = list(range(N_DD))
            # ddy = list(range(N_DD))
            ddx = [CB_Js, CB_Js, CB_Js, CB_lm_i/Js, CB_ls_i/CB_lm_i]
            ddy = [DF_data, DF_CF, (DF_CF-DF_data)/DF_data, DF_i, DF_i]
            for k in range(N_DD):
                description[k] = "%d%d%d; %s and %s; %s"%(j,i,k, fitmodel_name[j], combmodel_name[i], DD_name[k])
                # DD[j][i].append([ddx[k], ddy[k], description[k]]) #all did
                DD[j][i][k] = [ddx[k], ddy[k], description[k]]
        
        # plot
        PLOT = Plot_model_fit()
        dd_datafit = [DD[j][i][k] for k in range(2) for i in range(N_combmodel)] #[2*i+3 for i in range(5) for j in range(2)]
        dd_data = [DD[j][i][0] for i in range(N_combmodel)]
        dd_fit = [DD[j][i][1] for i in range(N_combmodel)]
        dd_er = [DD[j][i][2] for i in range(N_combmodel)]
        dd_lm = [DD[j][i][3] for i in range(N_combmodel)]
        dd_ls = [DD[j][i][4] for i in range(N_combmodel)]
        # dddd = [dd_data, dd_fit, dd_er, dd_lm, dd_ls]
        dddd = [dd_datafit, dd_er, dd_lm, dd_ls]
        st = "DFA_0old_"+fitmodel_name[j]
        xl = ["scaled comb_data and comb_fit", "scaled comb_data and comb_fit", "scaled locallyBlur mean of comb_data", "scaled locallyBlur standard/mean of comb_data"]
        yl = ["scaled log DF", "errors: (y_DF-y_fit)/y_DF", "scaled log DF", "scaled log DF"]
        PLOT.plot_actions_Comb_NDF_subplot(dddd, subtitle=st, xl=xl, yl=yl, lim=[True])
        # add.DEBUG_PRINT_V(0, j)
    '''

    

    '''
    ## comb 2 main and DF
    fitmodel = [
        gm.AA_fCombinationFixed_changedPowerLawDPL_log, 
        gm.AA_fCombinationSum_changedPowerLaw1_log, 
        gm.AA_fCombinationSum_changedPowerLaw2_log, 
        gm.AA_fCombinationSum_changedPowerLaw3_log, 
        gm.AA_fCombinationSum_usual1_log, 
        gm.AA_fCombinationFree_changedPowerLaw1_log, 
        gm.AA_fCombinationFree_changedPowerLaw2_log, 
        gm.AA_fCombinationFree_changedPowerLaw3_log, 
        gm.AA_fCombinationFree_usual1_log
    ]
    fitmodel_name = [
        "AA_fCombinationFixed_changedPowerLawDPL_log", 
        "AA_fCombinationSum_changedPowerLaw1_log", 
        "AA_fCombinationSum_changedPowerLaw2_log", 
        "AA_fCombinationSum_changedPowerLaw3_log", 
        "AA_fCombinationSum_usual1_log", 
        "AA_fCombinationFree_changedPowerLaw1_log", 
        "AA_fCombinationFree_changedPowerLaw2_log", 
        "AA_fCombinationFree_changedPowerLaw3_log", 
        "AA_fCombinationFree_usual1_log"
    ]
    N_fitmodel = len(fitmodel)
    fitmodel_params = list(range(N_fitmodel))
    fitmodel_params_name = [
        ["coef_total", "power_free_1", "power_free_2", "log_penalty"], 
        ["power_free_1", "power_free_2", "power_free_3", "log_penalty"], 
        ["power_free_1", "power_free_2", "power_free_3", "power_free_4", "log_penalty"], 
        ["power_free_1", "power_free_2", "power_free_3", "log_penalty"], 
        ["power_free_1", "power_free_2", "power_free_3", "power_Einasto", "log_penalty"], 
        
        ["coef_free_p1", "coef_free_p2", 
            "power_free_1", "power_free_2", "power_free_3", "log_penalty"], 
        ["coef_free_p1", "coef_free_p2", "coef_free_p3", "coef_free_p4", 
            "power_free_1", "power_free_2", "power_free_3", "power_free_4", "log_penalty"], 
        ["coef_free_p1", "coef_free_p2", 
            "power_free_1", "power_free_2", "power_free_3", "log_penalty"], 
        ["coef_free_p1", "coef_free_p2", "coef_free_p3", "coef_free_p4", 
            "power_free_1", "power_free_2", "power_free_3", "power_free_4", "power_Einasto", "log_penalty"]
    ]

    combmodel = [
        gm.AA_combination_freeCoef
    ]
    combmodel_name = [
        "gm.AA_combination_freeCoef"
    ]
    N_combmodel = len(combmodel)
    # combmodel_params = [[], [], [], ar, [], [0., 0., 0.], [0., 0., 0.], [0., 0., 0., 0., 0., 0.]]
    
    DD_name = ["", "", "", "", "", ""]

    AA_data = x #data
    DF_data = y #data
    # Js = Jes
    Js = gm.AA_combination_estimateScaleFreq(AA_data)
    CB = list(range(N_combmodel)) #data
    DF = list(range(N_fitmodel)) #data
    for k in range(len(DF)):
        DF[k] = []
    PM = list(range(N_fitmodel)) #params
    for k in range(len(PM)):
        PM[k] = []
    DD = [[[0 for k in range(len(DD_name))] for i in range(N_combmodel)] \
        for j in range(N_fitmodel)] #decrease dimension and plot

    #fitmodel
    for j in np.arange(N_fitmodel):
        #usual comb: -xcf -edms
        for i in np.arange(N_combmodel):
            print(j,i)

            nf = len(fitmodel_params_name[j])
            WF = Wrap_func(MG, fitmodel[j], nf)
            WF.set_bounds(fitmodel_params_name[j])
            ff = WF.funcfit
            # nf = WF.N_fitparams
            pe = np.array(WF.reference_list)
            CF = Curve_fit_galaxy(WF, nf, AA_data, DF_data, yerr, tag0=0, mf=5000)
            CF.run()
            # add.DEBUG_PRINT_V(1, "AA_fCombination_exp_E")
            P_CF, covariance, residual, residual_sigma, E_CF = CF.display()
            DF_CF = ff(AA_data, *P_CF)
            DF[j].append(DF_CF)
            PM[j].append(P_CF)

            # if j<=4:
            #     # CB[i] = gm.AA_combination_sumWeightFrequency_rateF1(AA_data)
            #     CB[i] = gm.AA_combination_sum(AA_data)
            # else:
            #     CB[i] = combmodel[i](AA_data, P_CF[0], P_CF[1])
            CB[i] = gm.AA_combination_sum(AA_data)

            Js0 = np.median(np.sum(x[:,0:3], axis=1)) #for log DF??
            CB_Js = CB[i]/Js
            cf_i = add.merge_array_by_hstack([CB_Js, DF_data])
            cf_lm_i, cf_ls_i, cf_f_sortIndex_i = add.neighbourAverage_bin(cf_i, 
                which_column=-1, N_neighbour=N_neighbour, is_all=1)
            CB_lm_i = cf_lm_i[:,0]
            CB_ls_i = cf_ls_i[:,0]#/cf_lm_i[:,0]
            DF_i = cf_lm_i[:,1]

            ddx = [CB_Js, CB_Js, CB_Js, CB_Js, CB_lm_i, CB_ls_i/CB_lm_i]
            ddy = [DF_data, DF_CF, (DF_CF-DF_data)/DF_data, DF_CF, DF_i, DF_i]
            description = list(range(len(ddx)))
            for k in range(len(ddx)):
                # add.DEBUG_PRINT_V(1, k)
                description[k] = "%d%d%d; %s and %s"%(j,i,k, fitmodel_name[j], combmodel_name[i])
                # DD[j][i].append([ddx[k], ddy[k], description[k]]) #all did
                DD[j][i][k] = [ddx[k], ddy[k], description[k]]
        
        # plot
        PLOT = Plot_model_fit()
        dd_datafit = [DD[j][i][k] for k in [0,1] for i in range(N_combmodel)] #[2*i+3 for i in range(5) for j in range(2)]
        dd_data = [DD[j][i][0] for i in range(N_combmodel)]
        dd_fit = [DD[j][i][1] for i in range(N_combmodel)]
        dd_er = [DD[j][i][2] for i in range(N_combmodel)]
        dd_lm = [DD[j][i][k] for k in [4,3] for i in range(N_combmodel)]
        dd_ls = [DD[j][i][5] for i in range(N_combmodel)]
        # dddd = [dd_data, dd_fit, dd_er, dd_lm, dd_ls]
        dddd = [dd_datafit, dd_er, dd_lm, dd_ls]
        # st = "%d_2__triaxial_composition_method__actions-Comb-NDF"%(j)
        st = "DFA_"+fitmodel_name[j]
        xl = [
            "scaled comb_data and comb_fit", 
            "scaled comb_data and comb_fit", 
            "scaled locallyBlur mean of comb_data", 
            "scaled locallyBlur standard/mean of comb_data"
        ]
        yl = [
            "scaled log DF", "errors: (y_DF-y_fit)/y_DF", 
            "scaled log DF", "scaled log DF"
        ]
        PLOT.plot_actions_Comb_NDF_subplot(dddd, subtitle=st, xl=xl, yl=yl, lim=[True])
        print("")
        # add.DEBUG_PRINT_V(0, j)



    ## with axisRatio, ds, rs, time power, common shape, nothing and gJ and colliSize
    a3 = 1
    '''



    '''
    # f = gm.AA_polysum_frequency_log
    # funcname = "nothing"
    # params_name = ["coef_free_n3", "coef_free_n2", "coef_free_n1", "coef_free_0", 
    #     "coef_free_p1", "coef_free_p2", "coef_free_p3", 
    #     "action_scale", "log_penalty"]
    # funcname = "AA_polysum_frequency_log"
    # params_name = ["coef_free_n3", "coef_free_n2", "coef_free_n1", "coef_free_0", 
    #     "coef_free_p1", "coef_free_p2", "coef_free_p3", 
    #     "action_scale", "log_penalty"]
    
    # f = gm.AA_powermult_frequency_log
    # funcname = "AA_powermult_frequency_log"
    # params_name = ["power_free_1", "power_free_2", "power_free_3", "power_free_4", "action_scale", "log_penalty"]
    
    # f = gm.AA_power1_coef_log
    # funcname = "AA_power1_coef_log"
    # params_name = ["power_free_1", "coef_exp_1", "coef_exp_2", "coef_exp_3", "coef_total", "log_penalty"]
    
    # f = gm.AA_exppower_coef_log
    # funcname = "AA_exppower_coef_log"
    # fc = gm.AA_combination_sumWeightCoef
    # fcn = "AA_combination_sumWeightCoef"
    # params_name = ["power_Einasto", "coef_exp_1", "coef_exp_2", "coef_exp_3", "coef_total", "log_penalty"]
    
    # f = gm.AA_exppower_frequency_log
    # funcname = "AA_exppower_frequency_log"
    # params_name = ["power_Einasto", "coef_total", "action_scale", "log_penalty"]

    # f = gm.AA_exp_frequency_log
    # funcname = "AA_exp_frequency_log"
    # params_name = ["coef_total", "action_scale", "log_penalty"]

    # f = gm.AA_exp_coef_log
    # funcname = "AA_exp_coef_log"
    # params_name = ["coef_free_p1", "coef_free_p2", "coef_free_p3", "coef_total", "log_penalty"]

    # f = gm.rho_doublepowerlaw_triaxial_log
    # funcname = "rho_doublepowerlaw_triaxial_log"
    # params_name = ["power1", "power2", "density_scale", "length_scale", 
    #             "axis_ratio_y", "axis_ratio_z", 
    #             "rotate_angle_x", "rotate_angle_y", "rotate_angle_z", 
    #             "log_penalty"]

    ff = gm.AA_P1Rpower_free_log
    funcname = "AA_P1Rpower_free_log"
    fc = gm.AA_combination_sumCoefPowerFree
    fcn = "AA_combination_sumCoefPowerFree"
    params_name = ["coef_free_p1", "coef_free_p2", "coef_free_p3", 
                "power_free_1", "power_free_2", "power_free_3", 
                "coef_total", "power_total", "log_penalty"]
    

    
    ## load
    nf = len(params_name)
    WF = Wrap_func(MG, ff, nf)
    WF.set_bounds(params_name)
    ff = WF.funcfit
    nf = WF.N_fitparams
    pe = np.array(WF.reference_list)



    ## firsthand data display
    # # COMB = add.norm_l(x[:,0:3], axis=1)
    # COMB = np.sum(x[:,0:3], axis=1)
    # plt.scatter(COMB, y)
    # plt.show()
    # plt.close()
    # exit(0)
    

    
    ## data and fit
    # data
    # x, y
    # xr, yr = RD.data_sample_combination(x, y, "radius")
    # xplot, yplot = xr, yr
    # dos0 = RD.data_original_simulationoOutput(snapshot_Id)
    # xv = dos0[:,0:6]
    # expect
    P_ER = pe
    # # MCMC
    # P_MC, E_MC = [0.9401185520037684, 3.211754159203118, 0.0008978270962895497, 
    #               23.894278631802607, -54.143804306504165], [0.0026085736587127986]
    # MF = MCMC_fit_galaxy(WF, WF.N_paramsfit, x, y, yerr, nw=60, ns=500,
    #                      nb=400, nth=1, nd=10, sd=250)
    # # MF = MCMC_fit_galaxy(WF, nf, x, y, yerr, nw=60, ns=5000,
    # #                      nb=4000, nth=10, nd=1000, sd=250)
    # MF.run()
    # P_MC, E_MC = MF.display(name=name, is_show=0)
    # P_MC = P_MC[0:-1]
    # CFLS
    CF = Curve_fit_galaxy(WF, nf, x, y, yerr, tag0=0, mf=5000)
    add.DEBUG_PRINT_V(1, "before CF.run()")
    CF.run()
    add.DEBUG_PRINT_V(1, "after CF.run()")
    P_CF, E_CF = CF.display()
    y_CF = ff(x, *P_CF) #params_fit
    
    ## plot and save
    is_show = 1
    id_relative_compare = 0
    xl = "xlabel"
    yl = "ylabel"
    zl = "zlabel"
    text = gm_name+"_"+funcname
    savename = gm_name+"_"+funcname
    # xl = r"radius: $r (\mathrm{kpc})$"
    # yl = r"logrithmic mass density: $\log(\frac{\rho(r)}{\mathrm{1e10\,\, M_{sun}\
    #     \,\, kpc^-3}})\,\, (\times 1)$"
    # text = name+", the curve_fit result: \n"\
    #     +r"$n_\mathrm{Einasto}=%e$, "%(P_CF[0])+"\n"\
    #     +r"$\rho_0=%e$, "%(P_CF[1])+"\n"\
    #     +r"$r_0=%e$"%(P_CF[2]) #"snapshot_%03d"%(snapshot_Id) #for+
    PLOT = Plot_model_fit()

    # dl_ER = [x, y, xerr, yerr, xplot, yplot, ff, P_ER, 0., "expected function"]
    # dl_MC = [x, y, xerr, yerr, xplot, yplot, WF.funcfit, P_MC, E_MC, "fit by MCMC"]
    # dl_CF = [x, y, xerr, yerr, xplot, yplot,
    #          WF.funcfit, P_CF, E_CF, "fit by curve_fit"]
    # dl = [dl_CF]  # dl_DG, dl_MC
    # PLOT.load(*dl)
    # PLOT.plot(name=name, xl=xl, yl=r"relative error of "+yl, is_relative=1,
    #           id_relative_compare=id_relative_compare, is_show=0)

    dd_CF = np.hstack(( np.array([ x[:,0] ]).T, np.array([ x[:,1]+x[:,2] ]).T, 
        np.array([ y_CF ]).T ))
    ddl_CF = [dd_CF, "curve_fit"]

    dd_da0 = np.hstack(( np.array([ x[:,0] ]).T, np.array([ x[:,1]+x[:,2] ]).T, 
        np.array([ y ]).T ))
    ddl_da0 = [dd_da0, "data0"]
    dd_da1 = np.hstack(( np.array([ x[:,1] ]).T, np.array([ x[:,0]+x[:,2] ]).T, 
        np.array([ y ]).T ))
    ddl_da1 = [dd_da1, "data1"]
    dd_da2 = np.hstack(( np.array([ x[:,2] ]).T, np.array([ x[:,0]+x[:,1] ]).T, 
        np.array([ y ]).T ))
    ddl_da2 = [dd_da2, "data2"]

    Ja = np.array([ x[:,0] ]).T
    Jb = np.array([ x[:,1] ]).T
    Jc = np.array([ x[:,2] ]).T
    fJ = np.array([ y ]).T
    dd_da = np.hstack(( 1./(1.+Ja), 1./(1.+Jb+Jc), fJ ))
    ddl_da = [dd_da, "data"]

    # ddl = [ddl_da]
    # ddl = [ddl_da0, ddl_da1, ddl_da2]
    ddl = [ddl_da, ddl_CF]
    # PLOT.plot_x_scatter3d_dd(ddl, nm=savename, text=text, k_median=6., is_show=is_show)



    ## a kind of law model
    N_neighbour = 32
    fitmodel = []
    fitmodel_name = []
    combmodel = []
    combmodel_name = []
    # J3
    # Cb
    # NJ
    #() plot
    #() remove bad code
    func02_j = ff
    func12_j = gm.NDF_combination_P1Rpower_free_log #rewrite
    func01_P_j = P_CF[0:-2] #_j
    func12_P_j = P_CF[-2:] #_j

    #1
    func01_i = fc
    func01_name_i = fcn
    comb_i = func01_i(x, *func01_P_j) *1e5

    cf_i = add.merge_array_by_hstack([comb_i, y])
    cf_la_i, cf_ls_i, cf_f_sortIndex_i = add.neighbourAverage_bin(cf_i, which_column=-1, N_neighbour=N_neighbour, is_all=1)
    Cb_la_i = cf_la_i[:,0]
    Cb_ls_i = cf_ls_i[:,0]
    NJ_i = cf_la_i[:,1]
    dd_f_1 = [comb_i, y_CF, func01_name_i+"; x:Comb(J3)*1eX, y:funcFit(J3) "]
    dd_d_1 = [comb_i, y, func01_name_i+"; x:Comb(J3)*1eX, y:NDF "]
    dd_e_1 = [comb_i, y_CF-y, func01_name_i+"; x:Comb(J3)*1eX, y:errorFit "]
    dd_a_1 = [Cb_la_i, NJ_i, func01_name_i+"; x:Comb(J3)*1eX, y:NDF; neighbourAverage "] #av: Cb NDF ??
    dd_s_1 = [Cb_ls_i, NJ_i, func01_name_i+"; x:Comb(J3)*1eX, y:NDF; neighbourScatter "]

    #2
    func01_i = gm.AA_combination_sumWeightFrequency_rateF1
    func01_name_i = "AA_combination_sumWeightFrequency_rateF1"
    comb_i = func01_i(x) *1e0

    cf_i = add.merge_array_by_hstack([comb_i, y])
    cf_la_i, cf_ls_i, cf_f_sortIndex_i = add.neighbourAverage_bin(cf_i, which_column=-1, N_neighbour=N_neighbour, is_all=1)
    Cb_la_i = cf_la_i[:,0]
    Cb_ls_i = cf_ls_i[:,0]
    NJ_i = cf_la_i[:,1]
    dd_f_2 = [comb_i, y_CF, func01_name_i+"; x:Comb(J3), y:funcFit(J3)"]
    dd_d_2 = [comb_i, y, func01_name_i+"; x:Comb(J3), y:NDF"]
    dd_e_2 = [comb_i, y_CF-y, func01_name_i+"; x:Comb(J3), y:errorFit"]
    dd_a_2 = [Cb_la_i, NJ_i, func01_name_i+"; x:Comb(J3), y:NDF; neighbourAverage "]
    dd_s_2 = [Cb_ls_i, NJ_i, func01_name_i+"; x:Comb(J3), y:NDF; neighbourScatter "]

    #3
    func01_i = gm.AA_combination_sum
    func01_name_i = "AA_combination_sum"
    comb_i = func01_i(x) *1e0

    cf_i = add.merge_array_by_hstack([comb_i, y])
    cf_la_i, cf_ls_i, cf_f_sortIndex_i = add.neighbourAverage_bin(cf_i, which_column=-1, 
        N_neighbour=N_neighbour, is_all=1)
    Cb_la_i = cf_la_i[:,0]
    Cb_ls_i = cf_ls_i[:,0]
    NJ_i = cf_la_i[:,1]
    dd_f_3 = [comb_i, y_CF, func01_name_i+"; x:Comb(J3), y:funcFit(J3)"]
    dd_d_3 = [comb_i, y, func01_name_i+"; x:Comb(J3), y:NDF"]
    dd_e_3 = [comb_i, y_CF-y, func01_name_i+"; x:Comb(J3), y:errorFit"]
    dd_a_3 = [Cb_la_i, NJ_i, func01_name_i+"; x:Comb(J3), y:NDF; neighbourAverage "]
    dd_s_3 = [Cb_ls_i, NJ_i, func01_name_i+"; x:Comb(J3), y:NDF; neighbourScatter "]

    #4
    func01_i = gm.AA_combination_norml2
    func01_name_i = "AA_combination_norml2"
    comb_i = func01_i(x) *1e0

    cf_i = add.merge_array_by_hstack([comb_i, y])
    cf_la_i, cf_ls_i, cf_f_sortIndex_i = add.neighbourAverage_bin(cf_i, which_column=-1, N_neighbour=N_neighbour, is_all=1)
    Cb_la_i = cf_la_i[:,0]
    Cb_ls_i = cf_ls_i[:,0]
    NJ_i = cf_la_i[:,1]
    dd_f_4 = [comb_i, y_CF, func01_name_i+"; x:Comb(J3), y:funcFit(J3)"] #should add fa fs
    dd_d_4 = [comb_i, y, func01_name_i+"; x:Comb(J3), y:NDF"]
    dd_e_4 = [comb_i, y_CF-y, func01_name_i+"; x:Comb(J3), y:errorFit"]
    dd_a_4 = [Cb_la_i, NJ_i, func01_name_i+"; x:Comb(J3), y:NDF; neighbourAverage "]
    dd_s_4 = [Cb_ls_i, NJ_i, func01_name_i+"; x:Comb(J3), y:NDF; neighbourScatter "]



    ## plot comb~J
    ddl_original = [dd_d_4, dd_d_3, dd_d_2, dd_d_1, dd_f_1]
    # PLOT.plot_actions_Comb_NDF(ddl_original)
    ddl_la = [dd_a_4, dd_a_3, dd_a_2, dd_a_1, dd_f_1]
    # PLOT.plot_actions_Comb_NDF(ddl_la)
    # ddl_ls = [dd_e_4, dd_e_3, dd_e_2, dd_e_1] #should be correspond
    # PLOT.plot_actions_Comb_NDF(ddl_ls)
    ddl_ls = [dd_s_4, dd_s_3, dd_s_2, dd_s_1]
    # PLOT.plot_actions_Comb_NDF(ddl_ls)
    dddd = [ddl_original, ddl_la, ddl_ls]
    # PLOT.plot_actions_Comb_NDF_subplot(dddd, subtitle="spherical: actions-Comb-NDF")
    PLOT.plot_actions_Comb_NDF_subplot(dddd, subtitle="triaxial composition method: actions-Comb-NDF")

    # galaxymodel_name = "galaxy_general"+gmn+"/"
    # snapshot_begin = 0
    # snapshot_end = 300
    # N_snapshot = 1
    # snapshot_sequence = np.linspace(snapshot_begin, snapshot_end, N_snapshot+1)[0:-1].astype(int)
    # fit_s_p = np.zeros((N_snapshot, nf-1))
    # params_snapshot = list(range(N_snapshot))
    # ystd_snapshot = list(range(N_snapshot))
    # median_snapshot = list(range(N_snapshot))

    # for Id in np.arange(len(snapshot_sequence)):
    #     params_snapshot[Id], ystd_snapshot[Id], median_snapshot[Id] = \
    #         plot_one_snapshot(snapshot_sequence[Id], galaxymodel_name, MG, WF, \
    #         is_fit=1, projection_surface="xy")
    #     fit_s_p[Id] = np.array(params_snapshot[Id])
    # print(fit_s_p)
    # # save np.array to txt
    # plot_params_t(snapshot_sequence, fit_s_p, pe, gmn, params_name=params_name)


    
    #():: what to do:
    #() WTD = {
    #() other methods of actions when this comb and funcfit
    #() shrink relative comparation of std dispertion and relative error
    #() AA_Posti15 wrapper and frequency weight
    #() RVF-axisratioC3-densityradiusscale-galaxypower fig and why bad
    #() see NJ with energy and I2I3 to fJ
    #() count 1e6 is OK, donot do dynamic as before, only static potential fudge and dynamic each ID file actions

    #() better potential all
    #() better limits action move with integrate method
    #() frequencies in fudge formula
    #() remove 0. and inf for action data KDE
    #() galaxies ensemble params fit law when core and cusp and a brifty
    #() what and why NJ
    #() time dependant fJ in a long term galaxy evoluiton
    #() main diagnal relaxiton scale of collision as well as diffuiton/composedlengthes/finestructurerate
    
    #() what are actions when norm_l2 big scattering
    #() small mass orbits and energy
    
    #() fJ solver
    #() other manifold to describe motions, from individual to chaos
    #() }
    '''



    '''
    # # plot time
    # #galaxy
    # M = 137.
    # ls = 19.6
    # ds = 0.000735
    # # 0.000891 19.6
    # # 0.002251 19.600000
    # MG = Model_galaxy(M, ls, ds)
    # MG.set_value('power1',          np.array([1., 1., 0., 1e1]))
    # MG.set_value('power2',          np.array([3., 3., 1e-1, 1e2]))
    # # MG.set_value('power_Einasto',   np.array([1./1.7, 1./1.7, 0., np.inf]))
    # # MG.set_value('power_Einasto',   np.array([1./1.7, 1./1.7, 1e-2, 1e2]))
    # MG.set_value('power_Einasto',   np.array([1.7, 1.7, 1e-2, 1e2]))
    # # MG.set_value('power_Einasto',   np.array([1.7, 1.7, 0.33, 3.]))
    # MG.set_value('density_scale',   np.array([ds, ds, ds*1e-1, ds*1e1]))
    # MG.set_value('length_scale',    np.array([ls, ls, ls*1e-1, ls*1e1]))
    # # MG.set_value('density_scale',   np.array([ds, ds, 0., ds*1e3]))
    # # MG.set_value('length_scale',    np.array([ls, ls, 0., ls*1e3]))
    # Js = (galaxy_models.G*M*ls)**0.5
    # MG.set_value('action_scale',    np.array([Js, Js, Js*1e-1, Js*1e1]))
    # MG.set_value('log_penalty',     np.array([-10., -10., -100., 1.]))

    # # rho
    # # f = galaxy_models.rho_spherical_Einasto_polygamma_log
    # f = galaxy_models.rho_spherical_Sersic_log
    # params_name = ['power_Einasto', 'density_scale', 'length_scale', 'log_penalty']
    # nf = len(params_name)
    # WF = Wrap_func_rho_Einasto_nmlz(MG, f, nf)
    # WF.set_bounds(params_name)
    # fs = WF.funcfit

    # gmn = "_41_EinastoUsual"
    # galaxymodel_name = "galaxy_general"+gmn+"/"
    # pe = np.array(WF.reference_list)
    # snapshot_begin = 0
    # snapshot_end = 300
    # N_snapshot = 30
    # snapshot_sequence = np.linspace(snapshot_begin, snapshot_end, N_snapshot+1)[0:-1].astype(int)
    # fit_s_p = np.zeros((N_snapshot, nf-1))
    # params_snapshot = list(range(N_snapshot))
    # ystd_snapshot = list(range(N_snapshot))
    # median_snapshot = list(range(N_snapshot))

    # for Id in np.arange(len(snapshot_sequence)):
    #     params_snapshot[Id], ystd_snapshot[Id], median_snapshot[Id] = \
    #         plot_one_snapshot(snapshot_sequence[Id], galaxymodel_name, MG, WF, \
    #             is_fit=1, projection_surface="xy")
    #     fit_s_p[Id] = np.array(params_snapshot[Id])
    # print(fit_s_p)
    # # save np.array to txt
    # plot_params_t(snapshot_sequence, fit_s_p, pe, gmn, params_name=params_name)



    # # model NFW
    # M = 137.
    # ls = 19.6
    # ds = 0.000891
    # id_relative_compare = 1
    # name = "rho_PL2" # special
    # MG = Model_galaxy(M, ls, ds)
    # MG.set_value("power1",          np.array([1., 1., 1e-4, 1e1])) # special
    # MG.set_value("power2",          np.array([3., 3., 1e-1, 1e2]))
    # MG.set_value("density_scale",   np.array([ds, ds, ds*1e-1, ds*1e1]))
    # MG.set_value("length_scale",    np.array([ls, ls, ls*1e-1, ls*1e1]))
    # MG.set_value("axis_ratio_y",    np.array([1., 1., 0.1, 10.]))
    # MG.set_value("axis_ratio_z",    np.array([1., 1., 0.1, 10.]))
    # MG.set_value("rotate_angle_x",  np.array([0., 0., 0., 2*np.pi])) # np.pi
    # MG.set_value("rotate_angle_y",  np.array([0., 0., 0., 2*np.pi]))
    # MG.set_value("rotate_angle_z",  np.array([0., 0., 0., 2*np.pi]))
    # # MG.set_value("coef_total",      np.array([1., 1., 0.5, 2.]))
    # MG.set_value("log_penalty",     np.array([-10., -10., -100., 1.]))

    # # rho
    # f = galaxy_models.rho_doublepowerlaw_spherical_log # special
    # params_name = ["power1", "power2", "density_scale", "length_scale", 
    #             "axis_ratio_y", "axis_ratio_z", 
    #             "rotate_angle_x", "rotate_angle_y", "rotate_angle_z", 
    #             "log_penalty"] # special
    # nf = len(params_name)
    # WF = Wrap_func_rho_doublepowerlaw_triaxial(MG, f, nf) # special
    # WF.set_bounds(params_name) # special
    # fs = WF.funcfit

    # gmn = "_11_NFW_triaxial" # special
    # galaxymodel_name = "galaxy_general"+gmn+"/"
    # pe = np.array(WF.reference_list)
    # snapshot_begin  = 200
    # snapshot_end    = 220
    # N_snapshot      = 20
    # snapshot_sequence = np.linspace(snapshot_begin, snapshot_end, N_snapshot+1)[0:-1].astype(int)
    # fit_s_p = np.zeros((N_snapshot, nf-1))
    # params_snapshot = list(range(N_snapshot))
    # ystd_snapshot = list(range(N_snapshot))
    # median_snapshot = list(range(N_snapshot))

    # for Id in np.arange(len(snapshot_sequence)): #snapshot_sequence[Id]
    #     params_snapshot[Id], ystd_snapshot[Id], median_snapshot[Id] = \
    #         plot_one_snapshot(snapshot_sequence[Id], galaxymodel_name, MG, WF, \
    #         is_fit=0, projection_surface="yz", is_show=0)
    #     fit_s_p[Id] = np.array(params_snapshot[Id])
    # print(fit_s_p)
    # # save np.array to txt
    # # plot_params_t()





    # # model NFW
    # M = 137.
    # ls = 19.6
    # ds = 0.000891
    # id_relative_compare = 1
    # name = "rho_PL2"
    # PL = Model_galaxy(M, ls, ds)
    # PL.set_value('power1',          np.array([1., 1., 1e-4, 1e1]))
    # PL.set_value('power2',          np.array([3., 3., 1e-1, 1e2]))
    # PL.set_value('density_scale',   np.array([ds, ds, ds*1e-1, ds*1e1]))
    # PL.set_value('length_scale',    np.array([ls, ls, ls*1e-1, ls*1e1]))
    # PL.set_value('log_penalty',     np.array([-10., -10., -100., 1.]))

    # # rho
    # f = galaxy_models.rho_doublepowerlaw_spherical_log
    # params_to_fit = {"power1": 1., "power2": 1.,
    #                  "density_scale": 1., "length_scale": 1., "log_penalty": 1.}
    # nf = len(params_to_fit)
    # WF = Wrap_func_rho_dsrs(PL, f, nf)
    # # WF = Wrap_func_rho_no_rs(PL,f,nf)
    # WF.set_bounds(params_name)
    # fs = WF.funcfit

    # galaxymodel_name = "galaxy_general_1_NFW/"
    # RD = Read_data_galaxy(PL, 2, gmn=galaxymodel_name)
    # dos = RD.data_secondhand_snapshot(0)
    # x, y = RD.data_sample_screen(dos, is_logy=True)
    # x, y = RD.data_sample_combination(x, y, "radius")
    # xplot, yplot = x, y
    # xerr = x*0.
    # yerr = y*0.1

    # # P_DG = np.array([1., 3., 1.*PL.params_dict["density_scale"]
    # #                 [1], 1.*PL.params_dict["length_scale"][1]])
    # P_DG = np.array(WF.reference_list)
    # dl_DG = [x, y, xerr, yerr, xplot, yplot, fs, P_DG, 0., "dice generating"]

    # # P_BS = np.array([0.01, 0.5, 0.5*PL.params_dict["density_scale"]
    # #                 [1], 1.*PL.params_dict["length_scale"][1]])
    # # dl_BS = [x, y, xerr, yerr, xplot, yplot, fs, P_BS, 0., "bounds setting"]

    # # P_MC = np.array([ 1.41293788,  2.63418101,  0.68387252, -1.61365958])
    # # P_MC = np.array([ 0.01012527,  7.32412133,  0.50011328, -1.2976536 ])
    # # P_MC = np.array([0.65603186, 0.66139846*5e0, 0.42029787*5e-3, 0.05020911])
    # # P_MC = np.array([0.65603186, 0.66139846, 0.42029787, 0.05020911])
    # # P_MC = np.array([8.64016914e-01,  2.96834992e+00,  8.72543147e-02, -7.77444384e+28]) #no limit
    # # P_MC = np.array([ 8.54115215e-01,  3.11961611e+00,  1.34192442e-03, -5.85229907e+61]) #>0 >0
    # # P_MC,E_MC = np.array([ 8.23958460e-01,  3.09479883e+00,  1.52617811e-03,  1.84392322e+01, -2.52587576e+02]), [0.0026366290195369404] #4+1
    # # P_MC,E_MC = [ 7.84924453e-01  3.07062478e+00  1.74479956e-03  1.72880528e+01 -2.46947095e+02] [0.0026413182201445503]
    # # P_MC, E_MC = np.array([8.80577330e-01,  3.14156998e+00,  1.18320265e-03,
    # #                       2.08244820e+01, -2.70017803e+02]), [0.0026255225199680646]
    # # P_MC, E_MC = [0.9401185520037684, 3.211754159203118, 0.0008978270962895497,
    # #               23.894278631802607, -54.143804306504165], [0.0026085736587127986]
    # # MF = MCMC_fit_galaxy(WF, nf, x, y, yerr, nw=60, ns=50,
    # #                      nb=40, nth=1, nd=1, sd=250)
    # P_MC, E_MC = [0.9401185520037684, 3.211754159203118, 0.0008978270962895497, 23.894278631802607, -54.143804306504165], [0.0026085736587127986]
    # # MF = MCMC_fit_galaxy(WF, nf, x, y, yerr, nw=60, ns=5000,
    # #                      nb=4000, nth=10, nd=1000, sd=250)
    # # MF.run()
    # # P_MC, E_MC = MF.display(name=name, is_show=0)
    # P_MC = P_MC[0:-1]
    # dl_MC = [x, y, xerr, yerr, xplot, yplot, fs, P_MC, E_MC, "fit by MCMC"]

    # # P_CF = np.array([0.01      , 6.72819878, 0.5       ])
    # # P_CF same as MC #>0 >0
    # P_CF, E_CF = np.array([9.66334946e-01, 3.23370892e+00, 8.01810586e-04,
    #                       2.51740591e+01]), [0.0026047347891023907]  # 4+1
    # CF = Curve_fit_galaxy(WF, nf, x, y, yerr, tag0=0, mf=5000)
    # CF.run()
    # P_CF, E_CF = CF.display()
    # dl_CF = [x, y, xerr, yerr, xplot, yplot,
    #          fs, P_CF, E_CF, "fit by curve_fit"]

    # dl = [dl_DG, dl_CF, dl_MC]  # , dl_BS
    # PLOT = Plot_model_fit()
    # PLOT.load(*dl)
    # PLOT.plot(name=name, is_relative=0, is_show=0)
    # PLOT.plot(name=name, is_relative=1,
    #           id_relative_compare=id_relative_compare, is_show=0)
    # xs, ys = RD.data_sample_screen(dos, is_logy=True)
    # plot_x_scatter(xs, name=name)
    # print("\n\n\n")

    # # fJ
    # name = "fJ_PL2"
    # Js = (galaxy_models.G*M*ls)**0.5
    # PL.set_value('action_scale',     np.array([Js, Js, Js*1e-1, Js*1e1]))

    # f = galaxy_models.AA3_spherical_pl2_Posti15_actionscale_log
    # params_name = ["power1", "power2", "action_scale", "log_penalty"]
    # nf = len(params_name)
    # WF = Wrap_func(PL, f, nf)
    # WF = Wrap_func_fJ_sph_sph_2pl(PL, f, nf)
    # WF.set_bounds(params_name)
    # fs = WF.funcfit

    # galaxymodel_name = "galaxy_general_1_NFW/"
    # RD = Read_data_galaxy(PL, 5, gmn=galaxymodel_name)
    # dos = RD.data_secondhand_snapshot(0)
    # x, y = RD.data_sample_screen(
    #     dos, x_down=0.02, x_up=20000., is_abs=1, is_logy=True)
    # xplot, yplot = RD.data_sample_combination(x, y, "radius")
    # # add.DEBUG_PRINT_V(0,np.median(x,axis=0),np.median(y))
    # xerr = x*0.
    # yerr = y*0.1

    # P_DG = np.array(WF.reference_list)
    # dl_DG = [x, y, xerr, yerr, xplot, yplot, fs, P_DG, 0., "expect"]

    # # # P_BS = np.array([1., 10., 2.*ds])
    # # P_BS = np.array([0.01, 0.5, 0.5*PL.params_dict["density_scale"]
    # #                 [1], 1.*PL.params_dict["length_scale"][1]])
    # # dl_BS = [x, y, xerr, yerr, fs, P_BS, 0., "bounds setting"]

    # # P_MC, E_MC = np.array([8.80577330e-01,  3.14156998e+00,  1.18320265e-03,
    # #                       2.08244820e+01, -2.70017803e+02]), [0.0026255225199680646]
    # # MF = MCMC_fit_galaxy(WF, nf, x, y, yerr, nw=60, ns=50,
    # #                      nb=40, nth=1, nd=1, sd=250)
    # P_MC, E_MC = np.array([0.10436638662871348, 2.608849878816311, 8015.229460199964,
    #                        -254.1406753266851]), 1.
    # # MF = MCMC_fit_galaxy(WF, nf, x, y, yerr, nw=60, ns=5000,
    # #                      nb=4000, nth=1000, nd=300, sd=250)
    # # MF.run()
    # # P_MC, E_MC = MF.display(name=name, is_show=0)
    # P_MC = P_MC[0:-1]
    # dl_MC = [x, y, xerr, yerr, xplot, yplot, fs, P_MC, E_MC, "fit by MCMC"]

    # # P_CF = np.array([0.01      , 6.72819878, 0.5       ])
    # # P_CF same as MC #>0 >0
    # # P_CF, E_CF = curve fit result:  [1.0000000e-03 2.6184198e+00 7.7782975e+03] [0.005728536186603672]
    # CF = Curve_fit_galaxy(WF, nf, x, y, yerr, tag0=0, mf=5000)
    # CF.run()
    # P_CF, E_CF = CF.display()
    # dl_CF = [x, y, xerr, yerr, xplot, yplot,
    #          fs, P_CF, E_CF, "fit by curve_fit"]

    # dl = [dl_DG, dl_CF, dl_MC]  # , dl_BS
    # PLOT = Plot_model_fit()
    # PLOT.load(*dl)
    # PLOT.plot(name=name, is_relative=0, is_show=0)
    # PLOT.plot(name=name, is_relative=1,
    #           id_relative_compare=id_relative_compare, is_show=0)
    # print("\n\n\n")



    # ## name
    # gmn = "_1_NFW"
    # # gmn = "_11_NFW_triaxial"
    # # gmn = "_41_EinastoUsual"
    # name = "rho_PL2" # special
    
    # ## model settings
    # M = 137.
    # ls = 19.6
    # ds = 0.004568 #0.000891
    # Js = (gm.G*M*ls)**0.5
    # id_relative_compare = 1
    # MG = Model_galaxy(M, ls, ds)
    # # MG.set_value("power1",          np.array([1., 1., 1e-3, 2.e0]))
    # # MG.set_value("power2",          np.array([3., 3., 2.e0, 1e1]))
    # MG.set_value("power1",          np.array([1., 1., 1e-4, 1e1]))
    # MG.set_value("power2",          np.array([3., 3., 2.e0, 1e2]))
    # MG.set_value("density_scale",   np.array([ds, ds, ds*1e-1, ds*1e1]))
    # MG.set_value("length_scale",    np.array([ls, ls, ls*1e-1, ls*1e1]))
    # MG.set_value("axis_ratio_y",    np.array([1., 1., 0.1, 10.]))
    # MG.set_value("axis_ratio_z",    np.array([1., 1., 0.1, 10.]))
    # MG.set_value("rotate_angle_x",  np.array([0., 2*np.pi, 0., 2*np.pi])) # np.pi #divide by zero
    # MG.set_value("rotate_angle_y",  np.array([0., 2*np.pi, 0., 2*np.pi]))
    # MG.set_value("rotate_angle_z",  np.array([0., 2*np.pi, 0., 2*np.pi]))
    # MG.set_value("log_penalty",     np.array([-10., -10., -100., 1.]))
    # # MG.set_value("coef_total",      np.array([1., 1., 0.5, 2.]))
    # MG.set_value("action_scale",    np.array([Js, Js, Js*1e-1, Js*1e1]))

    # ## func
    # f = gm.rho_doublepowerlaw_triaxial_log # special
    # params_name = ["power1", "power2", "density_scale", "length_scale", 
    #             "axis_ratio_y", "axis_ratio_z", 
    #             "rotate_angle_x", "rotate_angle_y", "rotate_angle_z", 
    #             "log_penalty"] # special
    # nf = len(params_name)
    # WF = Wrap_func(MG, f, nf)
    # WF.set_bounds(params_name)
    # fs = WF.funcfit

    # galaxymodel_name = "galaxy_general"+gmn+"/"
    # pe = np.array(WF.reference_list)
    # snapshot_begin = 0
    # snapshot_end = 300
    # N_snapshot = 1
    # snapshot_sequence = np.linspace(snapshot_begin, snapshot_end, N_snapshot+1)[0:-1].astype(int)
    # fit_s_p = np.zeros((N_snapshot, nf-1))
    # params_snapshot = list(range(N_snapshot))
    # ystd_snapshot = list(range(N_snapshot))
    # median_snapshot = list(range(N_snapshot))
    # # MCMC, 2hours, but wrang axis and angle: [1.5435289686055795, 3.1660126485084983, 0.004398944729028149, 17.535058841260998, 
    # # 0.9287093430241943, 1.1749753973657089, 4.394995914301586, 4.5719220355055645, 3.4928837257312453, -9.51827310119442]

    # for Id in np.arange(len(snapshot_sequence)):
    #     params_snapshot[Id], ystd_snapshot[Id], median_snapshot[Id] = \
    #         plot_one_snapshot(snapshot_sequence[Id], galaxymodel_name, MG, WF, \
    #             is_fit=1, projection_surface="xy")
    #     fit_s_p[Id] = np.array(params_snapshot[Id])
    # print(fit_s_p)
    # # save np.array to txt
    # plot_params_t(snapshot_sequence, fit_s_p, pe, gmn, params_name=params_name)

    # # fJ
    # name = "fJ_PL2"
    # Js = (galaxy_models.G*M*ls)**0.5
    # MG.set_value('action_scale',     np.array([Js, Js, Js*1e-1, Js*1e1]))

    # f = galaxy_models.AA3_spherical_pl2_Posti15_actionscale_log
    # params_name = ["power1", "power2", "action_scale", "log_penalty"]
    # nf = len(params_name)
    # WF = Wrap_func(MG, f, nf)
    # WF = Wrap_func_fJ_sph_sph_2pl(MG, f, nf)
    # WF.set_bounds(params_name)
    # fs = WF.funcfit

    # galaxymodel_name = "galaxy_general_11_NFW_triaxial/"
    # RD = Read_data_galaxy(MG, 5, gmn=galaxymodel_name)
    # dos = RD.data_secondhand_snapshot(0)
    # x, y = RD.data_sample_screen(
    #     dos, x_down=0.02, x_up=20000., is_abs=1, is_logy=True)
    # xplot, yplot = RD.data_sample_combination(x, y, "radius")
    # # add.DEBUG_PRINT_V(0,np.median(x,axis=0),np.median(y))
    # xerr = x*0.
    # yerr = y*0.1

    # P_DG = np.array(WF.reference_list)
    # dl_DG = [x, y, xerr, yerr, xplot, yplot, fs, P_DG, 0., "expect"]

    # # # P_BS = np.array([1., 10., 2.*ds])
    # # P_BS = np.array([0.01, 0.5, 0.5*PL.params_dict["density_scale"]
    # #                 [1], 1.*PL.params_dict["length_scale"][1]])
    # # dl_BS = [x, y, xerr, yerr, fs, P_BS, 0., "bounds setting"]

    # # P_MC, E_MC = np.array([8.80577330e-01,  3.14156998e+00,  1.18320265e-03,
    # #                       2.08244820e+01, -2.70017803e+02]), [0.0026255225199680646]
    # # MF = MCMC_fit_galaxy(WF, nf, x, y, yerr, nw=60, ns=50,
    # #                      nb=40, nth=1, nd=1, sd=250)
    # P_MC, E_MC = np.array([0.10436638662871348, 2.608849878816311, 8015.229460199964,
    #                        -254.1406753266851]), 1.
    # # MF = MCMC_fit_galaxy(WF, nf, x, y, yerr, nw=60, ns=5000,
    # #                      nb=4000, nth=1000, nd=300, sd=250)
    # # MF.run()
    # # P_MC, E_MC = MF.display(name=name, is_show=0)
    # P_MC = P_MC[0:-1]
    # dl_MC = [x, y, xerr, yerr, xplot, yplot, fs, P_MC, E_MC, "fit by MCMC"]

    # # P_CF = np.array([0.01      , 6.72819878, 0.5       ])
    # # P_CF same as MC #>0 >0
    # # P_CF, E_CF = curve fit result:  [1.0000000e-03 2.6184198e+00 7.7782975e+03] [0.005728536186603672]
    # CF = Curve_fit_galaxy(WF, nf, x, y, yerr, tag0=0, mf=5000)
    # CF.run()
    # P_CF, E_CF = CF.display()
    # dl_CF = [x, y, xerr, yerr, xplot, yplot,
    #          fs, P_CF, E_CF, "fit by curve_fit"]

    # dl = [dl_DG, dl_CF, dl_MC]  # , dl_BS
    # PLOT = Plot_model_fit()
    # PLOT.load(*dl)
    # PLOT.plot(name=name, is_relative=0, is_show=0)
    # PLOT.plot(name=name, is_relative=1,
    #           id_relative_compare=id_relative_compare, is_show=0)
    # print("\n\n\n")



    # # model EinastoUsual
    # M = 137.
    # ls = 19.6
    # ds = 0.001856
    # # 0.000891 19.6
    # # 0.002251 19.600000
    # EO = Model_galaxy(M, ls, ds)
    # EO.set_value('power1',          np.array([1., 1., 0., 1e1]))
    # EO.set_value('power2',          np.array([3., 3., 1e-1, 1e2]))
    # # EO.set_value('power_Einasto',   np.array([1./1.7, 1./1.7, 0., np.inf]))
    # # EO.set_value('power_Einasto',   np.array([1./1.7, 1./1.7, 1e-2, 1e2]))
    # EO.set_value('power_Einasto',   np.array([1.7, 1.7, 1e-2, 1e2]))
    # # EO.set_value('power_Einasto',   np.array([1.7, 1.7, 0.33, 3.]))
    # EO.set_value('density_scale',   np.array([ds, ds, ds*1e-1, ds*1e1]))
    # EO.set_value('length_scale',    np.array([ls, ls, ls*1e-1, ls*1e1]))
    # # EO.set_value('density_scale',   np.array([ds, ds, 0., ds*1e3]))
    # # EO.set_value('length_scale',    np.array([ls, ls, 0., ls*1e3]))
    # Js = (galaxy_models.G*M*ls)**0.5
    # EO.set_value('action_scale',    np.array([Js, Js, Js*1e-1, Js*1e1]))
    # EO.set_value('log_penalty',     np.array([-10., -10., -100., 1.]))

    # # rho
    # # f = galaxy_models.rho_spherical_Einasto_polygamma_log
    # f = galaxy_models.rho_spherical_Sersic_log
    # params_to_fit = [EO.params_dict['power_Einasto'], EO.params_dict['density_scale'],
    #                  EO.params_dict['length_scale'], EO.params_dict['log_penalty']]
    # nf = len(params_to_fit)
    # WF = Wrap_func_rho_Einasto_nmlz(EO, f, nf)
    # WF.set_bounds(params_name)
    # fs = WF.funcfit

    # # galaxymodel_name = "galaxy_general/"
    # galaxymodel_name = "galaxy_general_41_EinastoUsual/"
    # RD = Read_data_galaxy(EO, 2, gmn=galaxymodel_name)
    # dos = RD.data_secondhand_snapshot(000)
    # x, y = RD.data_sample_screen(dos, is_logy=True)
    # x, y = RD.data_sample_combination(x, y, "radius")
    # xplot, yplot = x, y
    # xerr = x*0.
    # yerr = y*0.1

    # # P_DG = [1.7, ds, ls*10]
    # P_DG = np.array(WF.reference_list)
    # dl_DG = [x, y, xerr, yerr, xplot, yplot, fs, P_DG, 0., "dice generating"]

    # # P_BS = np.array([0.01, 0.5, 0.5*PL.params_dict["density_scale"]
    # #                 [1], 1.*PL.params_dict["length_scale"][1]])
    # # dl_BS = [x, y, xerr, yerr, xplot, yplot, fs, P_BS, 0., "bounds setting"]

    # # P_MC, E_MC = [49.66144390504845, 0.013233587455570862, 33.86635087598731,
    # #               -101.71098414768285], [0.0029565396109290415]
    # # [2.6909641085656846, 0.013225781318987512, 32.5676226321689, -95.0819251849895] [0.00295804898835337]
    # # [1.8158939775877474, 0.013228523524488904, 31.9194008814398, -13.207124060010056] [0.0029599822369709393]
    # # MF = MCMC_fit_galaxy(WF,nf,x,y,yerr, nw=60,ns=50,nb=40,nth=1,nd=1,sd=250)
    # # P_MC, E_MC = [0.6060099218799953, 0.0008021382635978708,
    # #               146.2319542613304, -50.64079803790261], [0.002525518972712375]
    # P_MC, E_MC = [1.9404043921754777, 0.0007466010486617148, 20.558559903499933, -53.69901531172289], [0.00244509981316446]
    # # MF = MCMC_fit_galaxy(WF, nf, x, y, yerr, nw=60, ns=5000,
    # #                      nb=3000, nth=1, nd=200, sd=250)
    # # MF.run()
    # # P_MC, E_MC = MF.display(name=name, is_show=0)
    # P_MC = P_MC[0:-1]
    # dl_MC = [x, y, xerr, yerr, xplot, yplot, fs, P_MC, E_MC, "fit by MCMC"]

    # P_CF, E_CF = [1.69894278e+00, 1.32252673e-02,
    #               3.18001433e+01], [0.002962668707357795]
    # # [3.62735778e+00 1.32252673e-02 2.14697946e+01] [0.002962668700833189]
    # # [6.48419557e-01 1.32252673e-02 3.15808387e+01] [0.0029626687080339317]
    # CF = Curve_fit_galaxy(WF, nf, x, y, yerr, tag0=0, mf=5000)
    # CF.run()
    # P_CF, E_CF = CF.display()
    # dl_CF = [x, y, xerr, yerr, xplot, yplot,
    #          fs, P_CF, E_CF, "fit by curve_fit"]

    # id_relative_compare = 1
    # name = "rho_EinastoUsual"
    # xl = r"radius: $r (\mathrm{kpc})$"
    # yl = r"logrithmic mass density: $\log(\frac{\rho(r)}{\mathrm{1e10\,\, M_{sun}\,\, kpc^-3}})\,\, (\times 1)$"
    # dl = [dl_DG, dl_CF, dl_MC]  # dl_DG, dl_BS
    # PLOT = Plot_model_fit()
    # PLOT.load(*dl)
    # PLOT.plot(name=name, xl=xl, yl=yl, is_relative=0, is_show=0)
    # PLOT.plot(name=name, xl=xl, yl=r"relative error of "+yl, is_relative=1,
    #           id_relative_compare=id_relative_compare, is_show=0)
    # print("\n\n\n")

    # # fJ
    # name = "fJ_EinastoUsual"
    # f = galaxy_models.AA3_spherical_pl2_Posti15_actionscale_log
    # params_to_fit = [EO.params_dict['power1'], EO.params_dict['power2'],
    #                  EO.params_dict['action_scale'], EO.params_dict['log_penalty']]
    # nf = len(params_to_fit)
    # WF = Wrap_func_fJ_sph_sph_2pl(EO, f, nf)
    # WF.set_bounds(params_name)
    # fs = WF.funcfit

    # galaxymodel_name = "galaxy_general_41_EinastoUsual/"
    # RD = Read_data_galaxy(EO, 5, gmn=galaxymodel_name)
    # dos = RD.data_secondhand_snapshot(0)
    # x, y = RD.data_sample_screen(
    #     dos, x_down=0.02, x_up=20000., is_abs=1, is_logy=True)
    # xplot, yplot = RD.data_sample_combination(x, y, "radius")
    # # add.DEBUG_PRINT_V(0,np.median(x,axis=0),np.median(y))
    # xerr = x*0.
    # yerr = y*0.1

    # P_DG = np.array(WF.reference_list)
    # dl_DG = [x, y, xerr, yerr, xplot, yplot, fs, P_DG, 0., "analytical sph"]

    # # # P_BS = np.array([1., 10., 2.*ds])
    # # P_BS = np.array([0.01, 0.5, 0.5*PL.params_dict["density_scale"]
    # #                 [1], 1.*PL.params_dict["length_scale"][1]])
    # # dl_BS = [x, y, xerr, yerr, fs, P_BS, 0., "bounds setting"]

    # # P_MC, E_MC = np.array([8.80577330e-01,  3.14156998e+00,  1.18320265e-03,
    # #                       2.08244820e+01, -2.70017803e+02]), [0.0026255225199680646]
    # # P_MC, E_MC = np.array([0.10436638662871348, 2.608849878816311, 8015.229460199964,
    # #                        -254.1406753266851]), 1.
    # # MF = MCMC_fit_galaxy(WF,nf,x,y,yerr, nw=60,ns=50,nb=40,nth=1,nd=1,sd=250)
    # # P_MC, E_MC = [0.1571211257373524, 2.6283884549239054, 5168.563178529391, -53.18739168148305], [0.00784470115733137]
    # P_MC, E_MC = [0.21854577830976363, 2.862312361177838, 4228.718445674596, -53.02890712953341], [0.007275532676728827]
    # # MF = MCMC_fit_galaxy(WF, nf, x, y, yerr, nw=60, ns=5000,
    # #                      nb=3000, nth=1, nd=300, sd=250)
    # # MF.run()
    # # P_MC, E_MC = MF.display(name=name, is_show=0)
    # P_MC = P_MC[0:-1]
    # dl_MC = [x, y, xerr, yerr, xplot, yplot, fs, P_MC, E_MC, "fit by MCMC"]

    # # P_CF = np.array([0.01      , 6.72819878, 0.5       ])
    # # P_CF same as MC #>0 >0
    # # P_CF, E_CF = curve fit result:  [1.0000000e-03 2.6184198e+00 7.7782975e+03] [0.005728536186603672]
    # CF = Curve_fit_galaxy(WF, nf, x, y, yerr, tag0=0, mf=5000)
    # CF.run()
    # P_CF, E_CF = CF.display()
    # dl_CF = [x, y, xerr, yerr, xplot, yplot,
    #          fs, P_CF, E_CF, "fit by curve_fit"]

    # id_relative_compare = 1
    # name = "fJ_Einasto"
    # xl = r"action l2norm: $\sqrt{J_1^2+J_2^2+J_3^2}\,\, (\mathrm{kpc\,\, km/s})$"
    # yl = r"logrithmic probability density: $\log(\frac{f(J_1,J_2,J_3)}{\mathrm{1\,\,kpc^{-3}\,\, km/s^{-3}}})\,\, (\times 1)$"
    # dl = [dl_CF, dl_CF, dl_MC]  # dl_DG, dl_BS
    # PLOT = Plot_model_fit()
    # PLOT.load(*dl)
    # PLOT.plot(name=name, xl=xl, yl=yl, is_relative=0, is_show=0)
    # PLOT.plot(name=name, xl=xl, yl=r"relative error of "+yl, is_relative=1,
    #           id_relative_compare=id_relative_compare, is_show=0)
    # print("\n\n\n")





    # # record
    # # MCMC fit result:  [0.9401185520037684, 3.211754159203118, 0.0008978270962895497, 23.894278631802607, -54.143804306504165] [0.0026085736587127986]
    # # curve fit result:  [9.66334972e-01 3.23370895e+00 8.01810488e-04 2.51740606e+01] [0.002604734776795338]
    # # MCMC fit result:  [0.09310608337108237, 2.6036770765620822, 8003.736987626127, -57.21778399588263] [0.005746482676980818]
    # # curve fit result:  [1.0000000e-03 2.6184198e+00 7.7782975e+03] [0.005728536186603672]
    # # MCMC fit result:  [1.4162847498362068, 0.007028751182303079, 28.718170594223427, -50.889088854748564] [0.0024382237090563525]
    # # curve fit result:  [1.41748798e+00 7.03756581e-03 2.87127986e+01] [0.0024395812642415355]
    # # MCMC fit result:  [0.1571211257373524, 2.6283884549239054, 5168.563178529391, -53.18739168148305] [0.00784470115733137]
    # # curve fit result:  [7.43337513e-16 2.63639877e+00 4.97867349e+03] [0.007839740737250366]
    '''
