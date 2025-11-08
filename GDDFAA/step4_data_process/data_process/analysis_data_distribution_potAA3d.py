#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# from scipy.interpolate import spline
from mpl_toolkits.mplot3d import Axes3D
import galaxy_models

G = galaxy_models.G
M_total = galaxy_models.M_total
r_scale = galaxy_models.r_scale

def norm_l(a,l=2):
    x=0
    for e in a:
        x += e**l
    return x**(1./l)

def is_Econdition(t, col_J0):
    jj = t[col_J0:col_J0+3]
    ##escaper (one group, no compare)
    if jj[0]==0 and jj[2]==0 and jj[1]==0:
        return 0
    else:
        return 1

def is_Jcondition(t, col_J0):
    jj = t[col_J0:col_J0+3]
    ##J
    if (jj[0]==0 or jj[2]==0) and jj[1]!=0: #J
        return 0
    else:
        return 1

def is_Xcondition(t, col_x0):
    xx = t[col_x0:col_x0+6]
    rr = norm_l(xx[0:3])
    RR = norm_l(xx[0:2])
    zz = xx[2]
    vv = norm_l(xx[3:6])
    ##remove superhigh speed #nearly (74088.-46656.)/74088. = 0.37026239067055394 removed #but high speed particles has more none-zero J??
    if vv>=1562.5:
        return 0

    ##xv
    # if (rr>=0.1 and rr<=312.5*2) and (vv>0.1 and vv<=312.5*2):
    ##: rate torange #0.1*5^n: 0.6+~0.8+; rate to total #1: #~0.22 #20: #~0.22 #100: #~0.451 #312.5: #~0.455 #312.5*1.74: #~0.86
    # if (RR>=0.1 and RR<=312.5*2) and (zz>0.1 and zz<=312.5*2):
    ##: plot Rz: R, z seems not different, and seems no distingrish without V
    ##: plot vRvz: the data is too less
    if 1:
    # xx[3] = abs(xx[3])
    # if xx[3]>=0.0 and xx[3]<=1.: #disk: too less
    # if xx[3]>=1. and xx[3]<=10.: #disk: 0.3+
    # if xx[3]>=10. and xx[3]<=50.: #disk: 0.3+
    # if xx[3]>=50. and xx[3]<=350.: #disk: too less
    # if xx[3]>=350. and xx[3]<=1350.: #disk: no points
        return 1
    else:
        return 0

def remove_escaper(A, col_J):

    ##vector only
    if len(A.shape)!=2:
        print("analysis_data_distribution.py: plot_all_DF(): Input argument 1 should be a 2d-array!")
        return A
    if not isinstance(col_J, int):
        print("analysis_data_distribution.py: plot_all_DF(): Input argument 2 should be an int!")
        return A
    ##this is to judge the reliable data for extra aim and store the id in original data list
    Ntotal = len(A[:,0]) #the outest dim
    Ajudge = np.zeros(Ntotal)
    for i in range(Ntotal):
        t = A[i,:]
        if is_Econdition(t, col_J):
            Ajudge[i] = 1
    N = int(sum(Ajudge))
    if N==0:
        print("All are escaped!\n")
        return A
    ##rewrite A that not escaped
    Areliable = np.zeros((N, A.shape[1]))
    idx = 0
    for i in range(Ntotal):
        if Ajudge[i]:
            Areliable[idx] = A[i]
            idx += 1
    rate = float(N)/float(Ntotal)
    print(A.shape)
    print(r"The rate of non-escaped data A (len(A)=%d, idx=%d): %f = (float)(%d/%d)." % (Ntotal, idx, rate, N, Ntotal))
    return Areliable



def divided_bins_rho(J0, nmesh = 100, whatbin = 0, param1 = 0, param2 = 0, param3 = 0, param4 = 0):

    ##vector only
    L = len(J0.shape)
    if L!=1:
        print("analysis_data_distribution.py: plot_DF(): Input argument should be a vector!")
        return 0

    ##this is to judge the reliable data for extra aim and store the id in original data list
    N = len(J0) #the outest dim
    Jjudge = np.zeros(N)
    for i in range(N):
        t = J0[i]
        if 1: #aim none
        # if t>0: #aim 0
        # if t[0]>1.e-2 and t[1]>1.e-6 and t[2]>1.e-6 and t[0]<1.e8 and t[1]<1.e8 and t[2]<1.e8: #aim 1
            Jjudge[i] = 1
    N_eff = int(sum(Jjudge))

    ##resore the reliable data
    Jreliable = np.zeros(N_eff) #np.zeros((N_eff,3))
    idx = 0
    for i in range(N):
        if Jjudge[i]:
            Jreliable[idx] = J0[i]
            idx += 1
    rate = float(N_eff)/float(N)
    print("The rate of reliable data A: %f = (float)(%d/%d)." % (rate, N_eff, N))

    ##classifiy bins
    J = Jreliable #Jreliable[:,k]
    Jmax = max(J)
    Jmedian = np.median(J) #as scale
    # Jmesh = np.logspace(-2, np.log(max(J)), nmesh) #for log #*2.7/10*0.5
    Jmesh = np.linspace(0., Jmedian*5, nmesh) #for line
    # Jmesh = np.linspace(0., max(J), nmesh) #for line
    # Jmesh = np.logspace( np.log(min(J)), np.log(max(J)) ,nmesh ) #for log
    Jbin = (Jmesh[-1]-Jmesh[0])/nmesh #for log
    # Jbin = (Jmesh[-1]-Jmesh[0])/nmesh #for line
    print(r"data A: min max median bin_linear = %f %f %f %f" % (min(J), max(J), Jmedian, Jbin))

    ##ship to each bin
    Jdistributionfunction = np.zeros(nmesh)
    for j in J:
        for n in range(nmesh-1):
            if j>=Jmesh[n] and j<Jmesh[n+1]+1e-20: #remove zeros before
                Jdistributionfunction[n] += 1
    JdistributionOne = Jdistributionfunction/N_eff
    Jdistributionfunction = JdistributionOne/Jbin
    # print("rate in each bin: ", JdistributionOne)
    print("eff-sum: ", sum(JdistributionOne))

    ##return
    if whatbin==1:
        return Jmesh, Jdistributionfunction, np.array([Jmax, Jmedian, Jbin, N_eff])
    elif whatbin==2:
        modelID = param1
        rho_scale = param2
        cl = param3
        logJmesh = (Jmesh+(Jmesh[1]-Jmesh[0])/2)/r_scale #set to middle to avoid log(0)
        M_mesh = JdistributionOne *M_total #here N==N_eff
        rho_mesh = np.zeros(nmesh)
        V_mesh = np.zeros(nmesh)
        log_rho_mesh = np.zeros(nmesh)
        for nn in range(nmesh-1):
            V_mesh[nn] = 4./3*np.pi* (Jmesh[nn+1]**3 - Jmesh[nn]**3)
            rho_mesh[nn] = M_mesh[nn]/V_mesh[nn]
        log_rho_mesh = rho_mesh/rho_scale
        # print(JdistributionOne.shape, rho_mesh.shape, log_rho_mesh.shape)
        
        wy0 = np.where((log_rho_mesh!=0)) #&(y0<max(y0)) & (y0>min(y0)) & (y0!=np.NaN) & (y0!=np.Inf))
        logJmesh = logJmesh[wy0]
        log_rho_mesh = log_rho_mesh[wy0]
        print("mesh: ", Jmesh)
        print("rate in each bin: ", JdistributionOne)
        print("unzero-rate: ", float(len(log_rho_mesh))/nmesh)

        return logJmesh, log_rho_mesh, np.array([Jmax, Jmedian, len(logJmesh), N_eff])
    else:
        plot_DF(J0, nmesh = nmesh, axisscale = axisscale, param1 = param1, param2 = param2, param3 = param3, param4 = param4)
        return 1

def plot_DF(J0, nmesh = 100, axisscale = 0, param1 = 0, param2 = 0, param3 = 0, param4 = 0):

    ##vector only
    L = len(J0.shape)
    if L!=1:
        print("analysis_data_distribution.py: plot_DF(): Input argument should be a vector!")
        return 0

    ##this is to judge the reliable data for extra aim and store the id in original data list
    N = len(J0) #the outest dim
    Jjudge = np.zeros(N)
    for i in range(N):
        t = J0[i]
        if t>0: #aim none
        # if t[0]>1.e-2 and t[1]>1.e-6 and t[2]>1.e-6 and t[0]<1.e8 and t[1]<1.e8 and t[2]<1.e8: #aim 1
            Jjudge[i] = 1
    N_eff = int(sum(Jjudge))

    ##resore the reliable data
    Jreliable = np.zeros(N_eff) #np.zeros((N_eff,3))
    idx = 0
    for i in range(N):
        if Jjudge[i]:
            Jreliable[idx] = J0[i]
            idx += 1
    rate = float(N_eff)/float(N)
    print("The rate of reliable data A: %f = (float)(%d/%d)." % (rate, N_eff, N))

    ##classifiy bins
    J = Jreliable #Jreliable[:,k]
    Jmedian = np.median(J) #as scale
    Jmesh = np.logspace(0., np.log(max(J))*2.7/10*0.5, nmesh) #for log
    # Jmesh = np.linspace(0., Jmedian*5, nmesh) #for line
    # Jmesh = np.linspace(0., max(J), nmesh) #for line
    Jbin = (Jmesh[-1]-Jmesh[0])/nmesh #for line
    # Jmesh = np.logspace( np.log(min(J)), np.log(max(J)) ,nmesh ) #for log
    # Jbin = ... #for log
    print(r"data A: min max median bin_linear = %f %f %f %f" % (min(J), max(J), Jmedian, Jbin))

    ##ship to each bin
    Jdistributionfunction = np.zeros(nmesh)
    for j in J:
        for n in range(nmesh-1):
            if j>=Jmesh[n] and j<Jmesh[n+1]+1e-20: #remove zeros before
                Jdistributionfunction[n] += 1
    JdistributionOne = Jdistributionfunction/N_eff
    Jdistributionfunction = JdistributionOne/Jbin
    # Jmesh /= Jmedian
    print("rate in each bin: ", JdistributionOne)
    print("sum: ", sum(JdistributionOne))

    ##plot
    if axisscale==1:
        plt.scatter(Jmesh, Jdistributionfunction, s=2.)
        plt.xlabel(r'data A', fontsize=10)
        plt.ylabel(r'descrete DF(divided by bin) of data A', fontsize=10)
        plt.ylim(0., max(Jdistributionfunction)*1.2)
    else:
        modelID = param1
        rho_scale = param2
        cl = param3
        logJmesh = (Jmesh+(Jmesh[1]-Jmesh[0])/2)/r_scale #set to middle to avoid log(0)
        M_mesh = JdistributionOne *M_total/N #here N==N_eff
        rho_mesh = np.zeros(nmesh)
        V_mesh = np.zeros(nmesh)
        log_rho_mesh = np.zeros(nmesh)
        # print(Jdistributionfunction.shape, rho_mesh.shape, log_rho_mesh.shape)
        for nn in range(nmesh-1):
            V_mesh[nn] = 4./3*np.pi* (Jmesh[nn+1]**3 - Jmesh[nn]**3)
            rho_mesh = M_mesh/V_mesh
        log_rho_mesh = rho_mesh/rho_scale
        # print(Jdistributionfunction.shape, rho_mesh.shape, log_rho_mesh.shape)

        
        wy0 = np.where((log_rho_mesh!=0)) #&(y0<max(y0)) & (y0>min(y0)) & (y0!=np.NaN) & (y0!=np.Inf))
        logJmesh = logJmesh[wy0]
        log_rho_mesh = log_rho_mesh[wy0]
        sw = np.exp(np.linspace(10., 1., len(logJmesh)))/1000
        # plt.scatter(logJmesh, log_rho_mesh, color=cl, s=sw)
        plt.scatter(logJmesh[0:100], log_rho_mesh[0:100], color=cl, s=sw)
        # plt.plot(logJmesh[0:100], log_rho_mesh[0:100], color=cl)
        # plt.bar(logJmesh[0:100], log_rho_mesh[0:100], color=cl, width=2e-2)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r'scaled radical disdance $r/r_0$, $r_0=%f\,\mathrm{kpc}$'%(r_scale), fontsize=24)
        plt.ylabel(r'scaled distribution function of mass density (divided'
                    +'\n'+r' by bin) $\rho(r)/\rho_0$  ($\rho_0=%f\,\mathrm{\times 1e10M_\odot \cdot kpc^{-3}}$)'%(rho_scale), fontsize=24)

    plt.legend(fontsize=10)
    # plt.savefig("./data_process/0arrangement/analysis_data_DF__.png")
    # plt.show()
    return 1

def plot_all_DF(J0, col_J, col_x = 14, col_Pot = 23, nmesh = 100):

    ##vector only
    if len(J0.shape)!=2:
        print("analysis_data_distribution.py: plot_all_DF(): Input argument 1 should be a 2d-array!")
        return 0
    if not isinstance(col_J, int):
        print("analysis_data_distribution.py: plot_all_DF(): Input argument 2 should be an int!")
        return 0

    ##this is to judge the reliable data for extra aim and store the id in original data list
    Ntotal = len(J0[:,0]) #the outest dim
    Xjudge = np.zeros(Ntotal)
    Jjudge = np.zeros(Ntotal)
    for i in range(Ntotal):
        t = J0[i,:]
        # if 1: #aim none
        if is_Xcondition(t, col_x):
            Xjudge[i] = 1
            if is_Jcondition(t, col_J):
                Jjudge[i] = 1
    N = int(sum(Xjudge))
    N_eff = int(sum(Jjudge))
    if N_eff==0:
        print("No such J!\n")
        return 0

    ##resore the reliable data #J, rr, vv, Pot
    ps_rr = np.zeros(N)
    ps_vv = np.zeros(N)
    ps_RR = np.zeros(N)
    ps_zz = np.zeros(N)
    ps_vRR = np.zeros(N)
    ps_vzz = np.zeros(N)
    ps_Pot = np.zeros((N,2))
    ps_PotErr = np.zeros(N)
    ps_JIs = np.zeros(N)
    ps_not_JIs = np.ones(N)

    Jreliable = np.zeros(N_eff) #np.zeros((N_eff,3))
    idx = 0
    for i in range(N):

        ps_rr[i] = norm_l(J0[i, col_x:col_x+3])
        ps_vv[i] = norm_l(J0[i, col_x+3:col_x+6])
        ps_RR[i] = norm_l(J0[i, col_x:col_x+2])
        ps_zz[i] = J0[i, col_x+2]
        ps_vRR[i] = norm_l(J0[i, col_x+3:col_x+5])
        ps_vzz[i] = J0[i, col_x+6]
        # ps_Pot[i] = J0[i, col_Pot:col_Pot+2] #someone doesnot have pot
        # ps_PotErr[i] = (ps_Pot[i,1]-ps_Pot[i,0])/ps_Pot[i,0]

        if Jjudge[i]:
            Jreliable[idx] = J0[i, col_J]

            ps_JIs[idx] = 1
            ps_not_JIs[idx] = 0
            idx += 1
    rate = float(N_eff)/float(N)
    print(r"The rate of reliable data A (len(A)=%d, idx=%d): %f = (float)(%d/%d)." % (Ntotal, idx, rate, N_eff, N))

    ##classifiy bins
    J = Jreliable #Jreliable[:,k]
    Jmedian = np.median(J) #as scale
    Jmesh = np.linspace(0., max(J), nmesh) #for line
    Jbin = (Jmesh[-1]-Jmesh[0])/(nmesh-1) #for line
    # Jmesh = np.logspace( np.log(min(J)/Jmedian), np.log(max(J)/Jmedian) ,nmesh ) #for log
    # Jbin = ... #for log
    print(r"data A: min max median bin_linear = %f %f %f %f" % (min(J), max(J), Jmedian, Jbin))

    ##ship to each bin
    Jdistributionfunction = np.zeros(nmesh)
    for j in J:
        for n in range(nmesh-1):
            if j>=Jmesh[n] and j<Jmesh[n+1]: #+1e-20: #the first is >0
                Jdistributionfunction[n] += 1
    # Jmesh /= Jmedian #scaled
    # print(J)
    # print(Jdistributionfunction)
    JdistributionOne = Jdistributionfunction/N_eff
    Jdistributionfunction = JdistributionOne/Jbin

    # print("rate in each bin: ", JdistributionOne)
    print("sum of JdistributionOne: ", sum(JdistributionOne))

    ##plot DF of J_i
    # plt.subplot(2,2,k+1)
    # plt.axes(yscale = "log") #for log
    plt.scatter(Jmesh, Jdistributionfunction, s=2., label="rate=%f, bin=%f"%(rate, Jbin)) #, label="snapshot at %.1f(Gyr)"%(s/100))
    plt.xlabel(r'data A', fontsize=10)
    plt.ylabel(r'descrete DF(divided by bin) of data A', fontsize=10)
    plt.legend()
    # plt.ylim(0., max(Jdistributionfunction)*1.2)
    plt.savefig("./data_process/0arrangement/analysis_data_DF__.png")
    # plt.show()

    ##plot ps_JIs, ps_PotErr in param space
    PotMeanDiff = np.mean(ps_PotErr)
    # ps_rr = np.log(ps_rr)
    # ps_vv = np.log(ps_vv)
    ps_PotErr = ps_PotErr+1.
    ps_JIs = np.array(np.where(ps_JIs==0, np.nan, ps_JIs))
    ps_not_JIs = np.array(np.where(ps_not_JIs==0, np.nan, ps_not_JIs))

    fig = plt.figure(figsize=(16, 12), dpi=80, facecolor=(0.0, 0.0, 0.0))
    ax = Axes3D(fig)
    ax.grid(True) # ax.set_axis_off() #remove all relevent axis

    # ax.scatter(ps_vRR[:], ps_vzz[:], ps_PotErr[:], color="black", s=0.5, label="%f"%(PotMeanDiff))
    # ax.scatter(ps_vRR[:], ps_vzz[:], ps_JIs[:], color="red", s=0.5, label="%f"%(rate))
    # ax.scatter(ps_vRR[:], ps_vzz[:], ps_not_JIs[:], color="green", s=0.5, label="%f"%(rate))
    # # ax.scatter(ps_rr[:], ps_vv[:], po_esc_JIs[:], color="blue", s=0.8, label="%f"%(rate))

    # # ax.scatter(ps_RR[:], ps_zz[:], ps_PotErr[:], color="black", s=0.5, label="%f"%(PotMeanDiff))
    # ax.scatter(ps_RR[:], ps_zz[:], ps_JIs[:], color="red", s=0.5, label="%f"%(rate))
    # ax.scatter(ps_RR[:], ps_zz[:], ps_not_JIs[:], color="green", s=0.5, label="%f"%(rate))
    # # ax.scatter(ps_rr[:], ps_vv[:], po_esc_JIs[:], color="blue", s=0.8, label="%f"%(rate))

    ax.scatter(ps_rr[:], ps_vv[:], ps_PotErr[:], color="black", s=0.5, label="%f"%(PotMeanDiff))
    ax.scatter(ps_rr[:], ps_vv[:], ps_JIs[:], color="red", s=0.5, label="%f"%(rate))
    ax.scatter(ps_rr[:], ps_vv[:], ps_not_JIs[:], color="green", s=0.8, label="%f"%(rate))
    # ax.scatter(ps_rr[:], ps_vv[:], po_esc_JIs[:], color="blue", s=0.8, label="%f"%(rate))

    # ax.set_xlim(min(ps_rr), max(ps_rr))
    # ax.set_ylim(min(ps_vv), max(ps_vv))
    ax.set_zlim(0.5, 2.)
    # # ax.plot3D([-lim,lim],[0,0],[0,0], color="red", linewidth=0.5)
    # # ax.plot3D([0,0],[-lim,lim],[0,0], color="red", linewidth=0.5)
    # # ax.plot3D([0,0],[0,0],[-lim,lim], color="red", linewidth=0.5)
    # # # ax.arrow(-lim,0, 2*lim,0) #only 2d??
    # # ax.text3D(-10.,-10.,-10., r'O', fontsize=20)
    # ax.set_xlabel(r'$x\quad \mathrm{kpc}$', fontsize=20)
    # ax.set_ylabel(r'$y\quad \mathrm{km/s}$', fontsize=20)
    ax.set_xlabel(r'$R\quad \mathrm{kpc}$', fontsize=20)
    ax.set_ylabel(r'$z\quad \mathrm{kpc}$', fontsize=20)
    ax.set_zlabel(r'$rate\quad 1$', fontsize=20)
    ax.legend()
    ax.view_init(elev = 90., azim = 90.)
    plt.legend()
    plt.savefig("./data_process/0arrangement/analysis_data_ps__.png")
    plt.show()

    plt.close("all")
    print("Figs done.")
    return 1



if __name__ == '__main__':

    # #select #gas 0:2000, halo 2000:12000, disk 12000:17000, bulge 17000:18000 #ASF FP 0:3, TSF FP 3:6, ASF DP 7:10, TSF DP 10:13
    # filename = "./data_process/aa/20210314_Bovy13/Bovy_newdisk_1e4_pt2m0.action/Bovy13_newdisk_paramspace.action"
    # # filename = "./data_process/aa/20210419_NFWBovy13/allID.action"
    # # filename = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general_1_NFW/snaps/aa/allID_0.txt"
    # data = np.loadtxt(filename, dtype=float)
    # data_num = np.where(data==np.inf, 0., data)
    # data = np.array(data_num)
    # # print(data_num[30000][0], data[30000,0])

    # # filename2 = "./data_process/aa/20210314_Bovy13/Bovy_newdisk_1e4_pt2m0.action/Bovy13_newdisk.action"
    # filename2 = "./data_process/aa/20210419_NFWBovy13/allID.action"
    # data2 = np.loadtxt(filename2, dtype=float)
    # data_num2 = np.where(data2==np.inf, 0., data2)
    # data2 = np.array(data_num2)
    # # data2 = data2[0:2000, :]
    # # data2 = data2[2000:12000, :]
    # data2 = data2[12000:17000, :]
    # # data2 = data2[17000:18000, :]

    # ff = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general_1_NFW/snaps/aa/allID_0.txt"
    # data3 = np.loadtxt(ff, dtype=float)
    # data_num3 = np.where(data3==np.inf, 0., data3)
    # data3 = np.array(data_num3)
    # # data3 = data3[:, 7]
    # # pr = plot_DF(data3, nmesh = 100)





    ##plot by xv range
    # # init = 0.1
    # # increase = 5.
    # # gridx = 6
    # # gridv = 7
    # # i = 0
    # # for ivx in np.arange(gridv):
    # #     vx = init *increase**ivx
    # #     for ivy in np.arange(gridv):
    # #         vy = init *increase**ivy
    # #         for ivz in np.arange(gridv):
    # #             vz = init *increase**ivz

    # #             for ixx in np.arange(gridx):
    # #                 xx = init *increase**ixx
    # #                 for ixy in np.arange(gridx):
    # #                     xy = init *increase**ixy
    # #                     for ixz in np.arange(gridx):
    # #                         xz = init *increase**ixz
    # #                         xv0 = np.array([xx,xy,xz, vx,vy,vz])
    # #                         i += 1
    # # print(i)

    # # data_esc = remove_escaper(data, col_J=7)
    # # pr = plot_all_DF(data_esc, col_J=7, col_x=14, col_Pot=23, nmesh=100)
    # data_esc2 = remove_escaper(data2, col_J=7)
    # # pr = plot_all_DF(data_esc2, col_J=7, col_x=14, col_Pot=23, nmesh=100)
    # print(pr)
    # print(x0[0], v0[0])





    ##plot rho_r
    galbox = "../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galsbox/"
    cl = ["purple", "blue", "green", "yellow", "orange", "red"]
    model = ["", "", "", "", "",    "", "", "", "", ""]
    model[0] = "galaxy_general"
    model[1] = "galaxy_general_1_NFW_1e4"
    model[2] = "galaxy_general_2_Hernquist_1e4"
    model[3] = "galaxy_general_3_Burkert_center"
    model[4] = "galaxy_general_4_Einasto_center"
    model[5] = "galaxy_general_5_isothermal_center"
    model[6] = "galaxy_general_6_Plummer_center"
    funcs = [   galaxy_models.rho_spherical_cLB,                    
                galaxy_models.rho_spherical_scaled_NFW_DICE,        
                galaxy_models.rho_spherical_scaled_Hernquist_DICE,  
                galaxy_models.rho_spherical_scaled_Burkert_DICE,    
                galaxy_models.rho_spherical_scaled_Einasto_DICE,    
                galaxy_models.rho_spherical_scaled_isothermal_DICE, 
                galaxy_models.rho_spherical_scaled_Plummer_DICE     ]
    rho_scale = [0.001, 0.000854, 0.0033, 0.000854, 0.000854, 0.000854, 0.004393]

    snapshot = [0, 100, 200, 300]
    nmesh = 100
    modelID = 1
    for s in range(len(snapshot)):
        path_detail = "/snaps/aa/allID_%03d.txt" % (snapshot[s])
        filename = galbox+model[modelID]+path_detail
        dataread = np.loadtxt(filename, dtype=float)

        x = dataread[:, 0:3]
        l = len(x[:,0])
        r = np.zeros(l)
        for i in np.arange(0,l):
            r[i] = norm_l(x[i])
        A = r
        dataplot = divided_bins_rho(A, nmesh=nmesh, whatbin=2, param1=modelID, param2=rho_scale[modelID], param3=cl[s])
        print(dataplot)
        sw = np.exp(np.linspace(10., 1., len(dataplot[0])))*1e-3
        plt.scatter(dataplot[0], dataplot[1], color=cl[s], s=sw)
        plt.xscale("log")
        plt.yscale("log")

        if s==0:
            r = np.linspace(min(dataplot[0]), max(dataplot[0]), nmesh*10)
            rho_r_0 = funcs[modelID](r)
            plt.plot(r,rho_r_0)

    plt.show()