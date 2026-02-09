#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import galaxy_models
import analysis_data_distribution

if __name__ == '__main__':

    ## settings
    galbox = "/home/darkgaia/workroom/0prog/gadget/Gadget-2.0.7/"
    # galbox = "/home/darkgaia/0prog/gadget/gadget-2.0.7/"
    model = "galaxy_general/"
    # model = "galaxy_general_1_NFW"
    em = [0]
    # em = [0, 1, 2, 3]
    # data where
    col_xxx = 0
    col_JJJ = 9
    col_Omg = 6
    col_rho = -4
    col_fJ = -3
    col_H = -2
    col_HJ = -1

    cl = ["purple", "blue", "green", "orange", "red"]
    k_knn = 10
    datamin_x = 5.e-2  # 1.e-8
    datamax_x = 5.e2  # 1.e4
    datamin = 2.e-2
    datamax = 2.e4

    ## read data
    j_X00 = 0
    i_00X = 0
    snapid = j_X00*000 + i_00X
    postname = "snaps/aa_100/snapshot_%03d.secondhand_002.txt" % (snapid) #000
    # postname = "snaps/aa/snapshot_%03d.secondhand_002.txt" % (snapid)
    # postname = "snaps/aa/snapshot_%03d.secondhand_005.txt" % (snapid)
    filename = galbox+model+postname
    dataread = np.array(np.loadtxt(filename, dtype=float))
    # dataread = abs(dataread)
    print(filename)

    r_bins, rho_bins, D0, D \
        = analysis_data_distribution.divided_bins_1(dataread, colD=col_xxx, datarange="r",
                                                    nmesh=50, whatbin=1, datamin=datamin_x, datamax=datamax_x)  # whatbin=1,4

    xdata = np.hstack((dataread[:,col_JJJ:col_JJJ+3], dataread[:,col_Omg:col_Omg+3]))
    ydata = dataread[:,col_fJ]
    # xdata = xdata[0:1000]
    # ydata = ydata[0:1000]
    xdata_eff, xdata_listCondition,xdata_listConditionNot = analysis_data_distribution.screen_boundary(xdata, datamin,datamax)
    xdata_eff[:,1] = abs(xdata_eff[:,1]) #Lz
    xdata_eff = abs(xdata_eff)
    ydata_eff = ydata[xdata_listCondition]
    J_bins, fJ_bins, JD0, JD \
        = analysis_data_distribution.divided_bins_1(xdata_eff, colD=0, datarange="r",
        nmesh=50, whatbin=1, datamin=datamin, datamax=datamax)  # whatbin=1,4

    # r_bins_31, rho_bins_31, xx_points_inaline,fxx_points_inaline, xxx_points_inaline,fxxx_points_inaline, D1_datapoints,D2_datapoints \
    #     = analysis_data_distribution.divided_bins_123(dataread, colD=col_xxx,colD2=col_xxx, \
    #     nmesh=100,nmesh2=10,nmesh3=6, whatbin=1, datamin=datamin,datamax=datamax) #whatbin=1,4

    r_formula = np.logspace(np.log10(datamin), np.log10(datamax), 1000)
    # r_formula = np.logspace(-0.4, 3., 1000)
    # rho_formula = galaxy_models.rho_spherical_NFW(
    #     r_formula, galaxy_models.rho_NFWxxx, galaxy_models.r_scale)
    # rho_formula = galaxy_models.rho_spherical_doublepowerlaw(
    #     r_formula, galaxy_models.rho_NFWxxx, galaxy_models.r_scale, 1.09, 2.63)*1.4
    # rho_formula = galaxy_models.rho_spherical_doublepowerlaw(
    #     r_formula, galaxy_models.rho_NFWxxx, galaxy_models.r_scale, 1.09, 2.3)*1.4
    rho_formula = galaxy_models.rho_spherical_doublepowerlaw(
        r_formula, galaxy_models.rho_NFWxxx, galaxy_models.r_scale, 1.0, 3.6)*1.8



    r_kernel_sph = np.sqrt(dataread[:, col_xxx]**2+dataread[:, col_xxx+1]**2+dataread[:, col_xxx+2]**2)
    # r_kernel_sph = dataread[:, col_xxx]
    rho_kernel_sph = dataread[:, col_rho]

    J_kernel_sph = np.sqrt(dataread[:, col_JJJ]**2+dataread[:, col_JJJ+1]**2+dataread[:, col_JJJ+2]**2)
    # J_kernel_sph = dataread[:, col_JJJ]

    M0 = galaxy_models.M_total
    r0 = galaxy_models.r_scale
    J0 = galaxy_models.J0
    J_compose1 = dataread[:,col_JJJ]*dataread[:,col_Omg] \
        +dataread[:,col_JJJ+1]*dataread[:,col_Omg+1] +dataread[:,col_JJJ+2]*dataread[:,col_Omg+2]
    J_compose1 /= (dataread[:,col_Omg]*J0)
    J_compose1 = 1/J_compose1
    J_compose2 = dataread[:,col_JJJ] +dataread[:,col_JJJ+1] +dataread[:,col_JJJ+2]
    J_compose2 /= J0
    J_compose3 = (1+J_compose1)**1.67/(1+J_compose2)**2.9
    J_compose4 = np.exp(-J_kernel_sph/J0*5.)
    # J_compose5 = (1+J_compose1)**1.67
    J_compose5 = (1+J_compose2)**2.9
    # H = dataread[:,col_H]
    # HJ = dataread[:,col_HJ]
    # H = abs(H)
    # HJ = abs(HJ)
    fJ_kernel_sph = dataread[:,col_fJ]
    # fJ_kernel_sph *= J0

    ## scaled
    # fJ_bins *= galaxy_models.M_total
    # fJ_bins *= 10

    r_formula /= r0
    r_bins /= r0
    r_kernel_sph /= r0
    rho_formula /= (M0/r0**3)
    rho_bins /= (1./r0**3)
    rho_kernel_sph /= (M0/r0**3)

    J_bins /= J0
    J_kernel_sph /= J0
    fJ_bins *= J0**3
    fJ_kernel_sph *= J0**3



    ## plot
    szpp = 0.5
    szft = 20
    # plt.plot(fJ_kernel_sph,fJ_kernel_sph, color="k", label="line of y=x")
    # plt.plot(r_formula,np.zeros(len(r_formula)), color="k", label="line of y=0")

    plt.plot(r_formula, rho_formula, color="k", label="formula")
    # plt.scatter(r_formula,rho_formula, label=r"by formula", s=szpp)

    # plt.scatter(np.arange(len(r_kernel_sph)), np.sort(r_kernel_sph), label=r"debug", s=szpp)
    # plt.scatter(r_kernel_sph, H, label=r"H", s=szpp)
    # plt.scatter(J_kernel_sph, HJ, label=r"HJ", s=szpp)
    # plt.scatter(J_compose3, fJ_kernel_sph, label=r"x composed by J Posti(frequency-weight summation and average-weight summation)", s=szpp)
    # plt.scatter(J_compose4, fJ_kernel_sph, label=r"x composed by J L2-norm", s=szpp)
    # plt.scatter(J_compose5, fJ_kernel_sph, label=r"x composed by J summation", s=szpp)

    plt.scatter(r_kernel_sph, rho_kernel_sph, label=r"by kernel sph", s=szpp)
    # plt.scatter(J_kernel_sph, fJ_kernel_sph, label=r"by kernel sph", s=szpp)

    plt.scatter(r_bins, rho_bins, label=r"by bins", s=szpp*10)
    # plt.scatter(J_bins, fJ_bins, label=r"by bins", s=szpp*10)
    # plt.scatter(r_bins_31,rho_bins_31, label=r"by bins 31", s=szpp)

    plt.xscale("log")
    plt.yscale("log")
    # plt.xlim(datamin_x/r0, datamax_x/r0)
    # plt.xlim(datamin/J0, datamax/J0)
    # plt.ylim(0., 0.0001)
    # # plt.xlabel(r"dimensionless scaled radius $\~r=r/a_H$", fontsize=20)
    # # plt.ylabel(r"frequencies, setting $a_H=1\mathrm{kpc},\,v_H=1\mathrm{km/s}$", fontsize=20)
    plt.xlabel(r"scaled action composing", fontsize=szft)
    plt.ylabel(r"sclaed DF", fontsize=szft)
    plt.title(r"dentity distribution")
    plt.legend(fontsize=szft)
    plt.show()



    ## print
    # print("min r: ", min(r_formula), min(r_bins), min(r_kernel_sph))
    Ja = np.median(JD0[:,0])
    Jb = np.median(JD0[:,1])
    Jc = np.median(JD0[:,2])
    print("scales: ", J0, np.median(J_bins), Ja,Jb,Jc)
    JD0_idx0__l2norm = np.sqrt(JD0[:,0]**2 +JD0[:,1]**2 +JD0[:,2]**2)
    JD0_idx1__median_l2norm = np.sqrt(Ja**2 +Jb**2 +Jc**2)
    JD0_idx1__median_mult = Ja*Jb*Jc #1.606e10
    print(np.mean(JD0_idx0__l2norm), np.median(JD0_idx0__l2norm), JD0_idx1__median_l2norm, JD0_idx1__median_mult)
    print(np.mean(JD0[:,0]), np.mean(JD0[:,1]), np.mean(JD0[:,2]), np.mean(JD0))
    print(np.median(JD0[:,0]), np.median(JD0[:,1]), np.median(JD0[:,2]), np.median(JD0))
    print((np.mean(JD0[:,0])*np.mean(JD0[:,1])*np.mean(JD0[:,2]))/J0**3, \
        (np.median(JD0[:,0])*np.median(JD0[:,1])*np.median(JD0[:,2]))/J0**3)
    print((np.mean(JD0[:,0])*np.mean(JD0[:,1])*np.mean(JD0[:,2]))/np.mean(J_bins)**3, \
        (np.median(JD0[:,0])*np.median(JD0[:,1])*np.median(JD0[:,2]))/np.median(J_bins)**3)
    
    xabs = abs(dataread[:,col_xxx])
    rxxx = np.sqrt(dataread[:,col_xxx]**2 +dataread[:,col_xxx+1]**2 +dataread[:,col_xxx+2]**2)
    xcon = np.where(rxxx<160.,1,0)
    print(sum(xcon), np.mean(rxxx), np.median(rxxx), np.max(rxxx))
