#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import spline
from scipy.optimize import curve_fit

import galaxy_models
import analysis_data_distribution

if __name__ == '__main__':

    ## f1, f2, f3 of one halo
    galbox = "/home/darkgaia/0prog/gadget/gadget-2.0.7/"
    # galbox = "/home/darkgaia/0prog/gadget/gadget-2.0.7/galsbox/"
    # galbox = "/home/darkgaia/0prog/gadget/gadget-2.0.7/galsbox/gals_20210903_actions_spherical_1e4_centered/"
    # model = "galaxy_general" #centerno
    model = "galaxy_general_1_NFW"
    # model = "galaxy_general_2_Hernquist"
    # model = "galaxy_general_3_Burkert"
    # model = "galaxy_general_4_Einasto"
    # model = "galaxy_general_5_isothermal"
    # model = "galaxy_general_6_Plummer"
    ## snapshot/100
    em = [0]
    # em = [0, 1, 2, 3]
    ## under coordinates: spherical: 12 18, 45 51; axisymmetric: 21 27, 54 60; triaxial: 30 37, 63 70
    start = 45
    start2 = 51
    ## others
    cl = ["purple", "blue", "green", "orange", "red"]
    nmesh = 100
    k_knn = 10
    datamin = 1.e-8
    datamax = 1.e4 #1.e4

    ## DF
    for j_ss in range(len(em)):
        for i_00X in range(1):

            j_X00 = em[j_ss]
            snapid = j_X00*100 + i_00X
            ff = galbox+model+"/snaps/aa"
            filename = ff+"/allID_%03d.txt" % (snapid)
            print(filename)
            dataread = np.loadtxt(filename, dtype=float)
            print("max data: ", max(dataread[:,start]))
            x_points_inaline,fx_points_inaline, xx_points_inaline,fxx_points_inaline, xxx_points_inaline,fxxx_points_inaline, D1_datapoints,D2_datapoints \
                = analysis_data_distribution.divided_bins_123(dataread, colD=start,colD2=start2, \
                    nmesh=100,nmesh2=10,nmesh3=5, whatbin=1, datamin=datamin,datamax=datamax)
            # x_points_inaline, fx_points_inaline, xx_points_inaline, fxx_points_inaline, xxx_points_inaline, fxxx_points_inaline = analysis_data_distribution.divided_bins_123(dataread, colD=start, nmesh=100,nmesh2=10,nmesh3=5, whatbin=1)
            # x_points_inaline, fx_points_inaline, xx_points_inaline, fxx_points_inaline, xxx_points_inaline, fxxx_points_inaline = analysis_data_distribution.divided_bins_123(dataread, colD=start, nmesh=100,nmesh2=10,nmesh3=5, whatbin=5, param4=1.e4)
            #:: for F,D actions, whatbin=1, theta for 1, omega for max x selection (400. is too much, 1. is too less, 5.~20. is proper)
            #:: for actions-frequencies with interpolation at 45 51, curve_fit p0=[1.,3., 1e-2], max 1e4



    #         #1d distribution
    #         for k in range(3):
    #             plt.subplot(2,2,k+1)

    #             func = galaxy_models.func_powerlaw #func_WE15_1 #func_exp #func_poly5 #func_powerlaw_M1 #func_doublepowerlaw_down
    #             M0 = galaxy_models.M_total
    #             # J_scale = np.sqrt(galaxy_models.G*galaxy_models.M_total*galaxy_models.r_scale)
    #             x = x_points_inaline[:,k].T #to rescale, or it will have bad numerical value and stop to calculation
    #             y = fx_points_inaline[:,k].T
    #             # x *= J_scale
    #             # y /= scale
    #             # y1 /= scale
    #             # print("x: ", x)
    #             # print("y: ", y)
    #             x1 = np.linspace(min(x),max(x), 1000)

    #             # popt, pcov = f_J__fit(func, x, y)
    #             # y1 = func(x1, *popt) #a*(x+b)**(-c)+d
    #             # print("fit params: ", popt, pcov)
    #             # plt.plot(x1,y1, color=cl[j_ss], label="fJ1 of snapshot_%03d (0.01 Gyr/snapshot)" % (snapid))
    #             plt.plot(x,y, color=cl[j_ss], label="fJ1 of snapshot_%03d (0.01 Gyr/snapshot), particles count %d" % (snapid, len(D1_datapoints)))
    #             plt.legend()
    #             if k==0:
    #                 plt.xlabel(r'r-action $J_r\quad \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=10)
    #                 plt.ylabel(r'distribution function (divided by bin) $f_1(x1)', fontsize=10)
    #                 # plt.text(max(x)/5, max(y), r'$f_1 = a\times (J_R+b)^{-c} +d$'
    #                 #             +'\n'+r'$a=%.3g,\, b=%f\,\mathrm{kpc}\cdot\mathrm{km/s},\, c=%f,\, d=%f$'%(popt[0],popt[1],popt[2],popt[3]), fontsize=10)
    #             if k==1:
    #                 plt.xlabel(r'$\phi$-action $L_z\quad \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=10)
    #                 plt.ylabel(r'distribution function (divided by bin) $f_1(x2)', fontsize=10)
    #                 # plt.text(max(x)/5, max(y), r'$f_1 = a\times (L_z+b)^{-c} +d$'
    #                 #             +'\n'+r'$a=%.3g,\, b=%f\,\mathrm{kpc}\cdot\mathrm{km/s},\, c=%f,\, d=%f$'%(popt[0],popt[1],popt[2],popt[3]), fontsize=10)
    #             if k==2:
    #                 plt.xlabel(r'z-action $J_z\quad \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=10)
    #                 plt.ylabel(r'distribution function (divided by bin) $f_1(x3)', fontsize=10)
    #                 # plt.text(max(x)/5, max(y), r'$f_1 = a\times (J_z+b)^{-c} +d$'
    #                 #             +'\n'+r'$a=%.3g,\, b=%f\,\mathrm{kpc}\cdot\mathrm{km/s},\, c=%f,\, d=%f$'%(popt[0],popt[1],popt[2],popt[3]), fontsize=10)

    #             # if k==0:
    #             #     plt.xlabel(r'scaled R-action $J_R\quad \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=10)
    #             #     plt.ylabel(r'distribution function (divided by bin) $f_1(J_R)\, \times 1$'+'\n'+r'($L_z$ and $J_z$ are intergrated)', fontsize=10)
    #             #     # plt.text(max(x)/5, max(y), r'$f_1 = a\times (J_R+b)^{-c} +d$'
    #             #     #             +'\n'+r'$a=%.3g,\, b=%f\,\mathrm{kpc}\cdot\mathrm{km/s},\, c=%f,\, d=%f$'%(popt[0],popt[1],popt[2],popt[3]), fontsize=10)
    #             # if k==1:
    #             #     plt.xlabel(r'scaled $\phi$-action $L_z\quad \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=10)
    #             #     plt.ylabel(r'distribution function (divided by bin) $f_1(L_z)\, \times 1$'+'\n'+r'($J_R$ and $J_z$ are intergrated)', fontsize=10)
    #             #     # plt.text(max(x)/5, max(y), r'$f_1 = a\times (L_z+b)^{-c} +d$'
    #             #     #             +'\n'+r'$a=%.3g,\, b=%f\,\mathrm{kpc}\cdot\mathrm{km/s},\, c=%f,\, d=%f$'%(popt[0],popt[1],popt[2],popt[3]), fontsize=10)
    #             # if k==2:
    #             #     plt.xlabel(r'scaled z-action $J_z\quad \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=10)
    #             #     plt.ylabel(r'distribution function (divided by bin) $f_1(J_z)\, \times 1$'+'\n'+r'($J_R$ and $L_z$ are intergrated)', fontsize=10)
    #             #     # plt.text(max(x)/5, max(y), r'$f_1 = a\times (J_z+b)^{-c} +d$'
    #             #     #             +'\n'+r'$a=%.3g,\, b=%f\,\mathrm{kpc}\cdot\mathrm{km/s},\, c=%f,\, d=%f$'%(popt[0],popt[1],popt[2],popt[3]), fontsize=10)
    # plt.show()

            # ##2d distribution
            # nl = len(fJJ_all)
            # J_scale = 1e4
            # x_cellpoints = JJ_all/J_scale
            # y = fJJ_all*1e0
            # func = galaxy_models.func2_exp
            # popt, pcov = curve_fit(func, x_cellpoints, y, p0=[1e-10, 1., 1.])
            # yy = func(x_cellpoints, *popt)
            # sx = x_cellpoints.shape
            # sy = y.shape
            # syy = yy.shape
            # print("sum: ", sum(fJJ_all), sum(y), sum(yy), sum((y-yy)**2))
            # print(popt,pcov)
            # # x_cellpoints *= J_scale
            # # y /= scale
            # # yy /= scale
            # # print("at 00: ", x_cellpoints[0,0], y[0], yy[0])
            # # print(sx,sy,syy)
            # # print(x_cellpoints[:,0],x_cellpoints[:,1])
            # # print("y yy: ", y, yy)
            # # print("where 0: ", np.where(x_cellpoints[:,0]==0))
            # # print(np.where(x_cellpoints[:,1]==0))
            # # print(np.where(y==0))

            # fig = plt.figure(figsize=(16, 12), dpi=80, facecolor=(0.0, 0.0, 0.0))
            # ax = Axes3D(fig)
            # ax.grid(True) # ax.set_axis_off() #or to remove all relevent axis
            # nmeshmesh = len(J_discrete[:,0])
            # J1mesh = np.linspace(0., J_discrete[10,0], 2*nmeshmesh)*1e-4
            # J2mesh = np.linspace(0., J_discrete[10,1], 2*nmeshmesh)*1e-4
            # x_mesh, y_mesh = np.meshgrid(J1mesh, J2mesh, indexing='ij') #gen mesh is to product each other with dims
            # z_mesh = func2_shadow_exp(x_mesh, y_mesh, *popt)*1e0
            # ax.plot_surface(x_mesh, y_mesh, z_mesh, color="blue")
            # ax.scatter(x_cellpoints[:,0], x_cellpoints[:,1], yy[:], color="blue", s=0.1, label="fitted function value") #fitted function points
            # ax.scatter(x_cellpoints[:,0], x_cellpoints[:,1], y[:], color="red", s=5.0, label="bin-number data (%d points)"%(nl)) #data points
            # # ax.set_zlim(0., 1e-10)
            # ax.set_xlabel(r'scaled R-action $J_R\, \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=16)
            # ax.set_ylabel(r'scaled $\phi$-action $L_z\, \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=16)
            # ax.set_zlabel(r'scaled distribution function (divided by bin) $f_2(J_R,L_z)\, \times 1$'+'\n'+r'($J_z$ are intergrated)', fontsize=16)
            # ax.set_title(r'$f_2 = a\times\exp(-k_1 J_R -k_2 L_z)$'
            #                 +'\n'+r'$a=%.3g,\, k_1=%f\,\mathrm{kpc}^{-1}\cdot\mathrm{km/s}^{-1},\, k_2=%f\,\mathrm{kpc}^{-1}\cdot\mathrm{km/s}^{-1}$'%(popt[0],popt[1],popt[2]), fontsize=16)
            # ax.legend()
            # ax.view_init(elev = 0., azim = 90.)
            # # ax.set_xscale('log') #??
            # # ax.set_yscale('log')
            # plt.legend()
            # plt.show()



            # ##3d distribution
            # M0 = galaxy_models.M_total
            # J_scale = np.sqrt(galaxy_models.G*galaxy_models.M_total*galaxy_models.r_scale) #about 1e4 now
            # nl = len(fxxx_points_inaline)

            # # x_cellpoints = xxx_points_inaline
            # # y_cellpoints = fxxx_points_inaline
            # # func = galaxy_models.func3_exp
            # # popt, pcov = curve_fit(func, x_cellpoints, y_cellpoints, p0=[1e-10, 1., 1., 1.])
            # # func = galaxy_models.AA3_spherical_pl2_Posti15_simple
            # # popt, pcov = curve_fit(func, x_cellpoints, y_cellpoints, p0=[1.,3., 1.e9], maxfev=100000) #for spherical
            # # popt, pcov = curve_fit(func, x_cellpoints, y_cellpoints, p0=[1.,3., 1.e4]) #for tiaxial

            # x_cellpoints = xxx_points_inaline #the name: datapoints and mesh??
            # y_cellpoints = fxxx_points_inaline
            # Omg_cellpoints = analysis_data_distribution.fromAAAtoBBB_byinterpolation_atsomepoints(D1_datapoints,D2_datapoints,x_cellpoints,k=k_knn,funcname="gaussian") #interpolation
            # Omg_cellpoints = np.array( np.where(Omg_cellpoints==0., datamin, Omg_cellpoints) ) #remove zero
            # x_cellpoints_JOmg = np.array( np.hstack((x_cellpoints, Omg_cellpoints)) ) #horizontal merging to be as input argument
            # # print(x_cellpoints_JOmg)
            # # print(y_cellpoints)



            # ## curve_fit
            # func = galaxy_models.AA3_spherical_pl2_Posti15_interpolation
            # popt, pcov = curve_fit(func, x_cellpoints_JOmg, y_cellpoints, p0=[1.,3., 1e-2]) #for frequencies

            # yy = func(x_cellpoints_JOmg, *popt)
            # sx = x_cellpoints.shape
            # sy = y_cellpoints.shape
            # syy = yy.shape
            # # print(y_cellpoints)
            # print("optimize params", popt)
            # print("covariances of fitting: ", pcov)

            # fig = plt.figure(figsize=(16, 12), dpi=80, facecolor=(0.0, 0.0, 0.0))
            # ftsz = 24
            # ax = Axes3D(fig)
            # ax.grid(True) # ax.set_axis_off() #or to remove all relevent axis
            # ax.scatter(x_cellpoints[:,0], x_cellpoints[:,1], y_cellpoints[:], color="red", s=1.5, label="samples (%d points)"%(nl))
            # ax.scatter(x_cellpoints[:,0], x_cellpoints[:,1], yy[:], color="blue", s=1.5, label="fitted (%d points)"%(nl))
            # ax.plot([x_cellpoints[0,0], x_cellpoints[0,0]], [x_cellpoints[0,1], x_cellpoints[0,1]], [y_cellpoints[0], yy[0]], color="green", lw=0.5, 
            #             label="differences between by data points and fit value") #+'\n'+'(only 1/5 of these differences are displayed for clarity)
            # for i in np.arange(len(x_cellpoints[:,0])): # (7000,8000): 
            #     ax.plot([x_cellpoints[i,0], x_cellpoints[i,0]], [x_cellpoints[i,1], x_cellpoints[i,1]], [y_cellpoints[i], yy[i]], color="green", lw=0.5)
            # ax.set_xlabel(r'r-action $J_r\quad \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=ftsz)
            # ax.set_ylabel(r'$\phi$-action $L_z\quad \mathrm{kpc}\cdot\mathrm{km/s}$', fontsize=ftsz)
            # ax.set_zlabel(r'distribution function (divided by bin) $f(x_1,x_2,x_3)\, \times 1$'
            #                 +'\n'+r'(here only $x_1$ and $x_2$ axis displayed, $x_3$ axis are overlapped)', fontsize=ftsz)
            # # ax.set_title(r'$f = a\times\exp(-k_1 J_R -k_2 L_z -k_3 J_z)$'
            # #                 +'\n'+r'$a=%.3g,\, k_1=%f\,\mathrm{kpc}^{-1}\cdot\mathrm{km/s}^{-1},\, k_2=%f\,\mathrm{kpc}^{-1}\cdot\mathrm{km/s}^{-1},\, k_3=%f\,\mathrm{kpc}^{-1}\cdot\mathrm{km/s}^{-1}$'%(popt[0],popt[1],popt[2],popt[3]), fontsize=16)
            # ax.legend(fontsize=ftsz)
            # ax.view_init(elev = 0., azim = 180.)
            # plt.show()

            # ## MCMC fit
            # func = galaxy_models.AA3_spherical_pl2_Posti15_interpolation