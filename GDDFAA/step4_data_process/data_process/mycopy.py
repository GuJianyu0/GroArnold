#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

def norm_l(a,l=2):
    x=0
    for e in a:
        x += e**l
    return x**(1./l)

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
    vv = norm_l(xx[3:6])
    ##xv
    if rr>=0.1 and rr<=0.5:
    ##: rate torange #??
    ##: rate to total #1: #~0.22 #20: #~0.22 #100: #~0.451 #312.5: #~0.455 #312.5*1.74: #~0.86
    # if xx[3]>=0.1 and xx[3]<=30: #100.
        return 1
    else:
        return 0

def plot_DF(J0, nmesh = 100):

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
    print(J)
    Jmedian = np.median(J) #as scale
    Jmesh = np.linspace(0., max(J), nmesh) #for line
    Jbin = (Jmesh[-1]-Jmesh[0])/nmesh #for line
    # Jmesh = np.logspace( np.log(min(J)/Jmedian), np.log(max(J)/Jmedian) ,nmesh ) #for log
    # Jbin = ... #for log
    print(r"data A: min max median bin_linear = %f %f %f %f" % (min(J), max(J), Jmedian, Jbin))
    if Jbin==0:
        print("No such J!\n")
        return 0

    ##ship to each bin
    Jdistributionfunction = np.zeros(nmesh)
    for j in J:
        for n in range(nmesh-1):
            if j>Jmesh[n] and j<Jmesh[n+1]: #remove zeros before
                Jdistributionfunction[n] += 1
    JdistributionOne = Jdistributionfunction/N_eff
    Jdistributionfunction = JdistributionOne/Jbin
    # Jmesh /= Jmedian
    print("rate in each bin: ", JdistributionOne)
    print("sum: ", sum(JdistributionOne))

    ##plot
    # plt.subplot(2,2,k+1)
    # plt.axes(yscale = "log") #for log
    plt.scatter(Jmesh, Jdistributionfunction, s=2.) #, label="snapshot at %.1f(Gyr)"%(s/100))
    plt.xlabel(r'data A', fontsize=10)
    plt.ylabel(r'descrete DF(divided by bin) of data A', fontsize=10)
    plt.legend()
    plt.ylim(0., max(Jdistributionfunction)*1.2)
    plt.savefig("/home/darkgaia/0prog/data_process/0arrangement/analysis_data_DF__.png")
    plt.show()
    return 1

def plot_all_DF(J0, col_J, col_x = 14, nmesh = 100):

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

    ##resore the reliable data
    Jreliable = np.zeros(N_eff) #np.zeros((N_eff,3))
    idx = 0
    for i in range(N):
        if Jjudge[i]:
            Jreliable[idx] = J0[i, col_J]
            idx += 1
    rate = float(N_eff)/float(N)
    print("The rate of reliable data A: %f = (float)(%d/%d)." % (rate, N_eff, N))

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
            if j>=Jmesh[n] and j<Jmesh[n+1]+1e-20: #the first is >0
                Jdistributionfunction[n] += 1
    print(sum(Jdistributionfunction)/N_eff)
    # Jdistributionfunction /= Jmedian
    Jdistributionfunction /= (N_eff*Jbin)
    print("rate in each bin: ", Jdistributionfunction*Jbin)

    ##plot
    # plt.subplot(2,2,k+1)
    # plt.axes(yscale = "log") #for log
    plt.scatter(Jmesh, Jdistributionfunction, s=2., label="rate=%f, bin=%f"%(rate, Jbin)) #, label="snapshot at %.1f(Gyr)"%(s/100))
    plt.xlabel(r'data A', fontsize=10)
    plt.ylabel(r'descrete DF(divided by bin) of data A', fontsize=10)
    plt.legend()
    # plt.ylim(0., max(Jdistributionfunction)*1.2)
    plt.savefig("/home/darkgaia/0prog/data_process/0arrangement/analysis_data_DF__.png")
    plt.show()
    return 1



if __name__ == '__main__':

    # filename = "/home/darkgaia/0prog/data_process/aa/20210314_Bovy13/Bovy_newdisk_1e4_pt2m0.action/Bovy13_newdisk.action"
    filename = "/home/darkgaia/0prog/data_process/aa/20210314_Bovy13/Bovy_newdisk_1e4_pt2m0.action/Bovy13_newdisk_paramspace.action"
    #select #gas 0:2000, halo 2000:12000, disk 12000:17000, bulge 17000:18000 #ASF FP 0:3, TSF FP 3:6, ASF DP 7:10, TSF DP 10:13
    data = np.loadtxt(filename, dtype=float)
    data_num = np.where(data==np.inf, 0., data)
    data = np.array(data_num)
    print(data_num[30000][0], data[30000,0])



    # init = 0.1
    # increase = 5.
    # gridx = 6
    # gridv = 7
    # i = 0
    # for ivx in np.arange(gridv):
    #     vx = init *increase**ivx
    #     for ivy in np.arange(gridv):
    #         vy = init *increase**ivy
    #         for ivz in np.arange(gridv):
    #             vz = init *increase**ivz

    #             for ixx in np.arange(gridx):
    #                 xx = init *increase**ixx
    #                 for ixy in np.arange(gridx):
    #                     xy = init *increase**ixy
    #                     for ixz in np.arange(gridx):
    #                         xz = init *increase**ixz
    #                         xv0 = np.array([xx,xy,xz, vx,vy,vz])
    #                         i += 1
    # print(i)



    J0 = data[:, 0:3]
    x0 = data[:, 14:17]
    v0 = data[:, 17:20]
    # x = x0[0:2000, :]
    # x = x0[2000:12000, :]
    x = x0[12000:17000, :]
    # x = x0[17000:18000, :]
    # v = v0
    # J = J0

    l = len(x[:,0])
    r = np.zeros(l)
    for i in np.arange(0,l):
        r[i] = norm_l(x[i])
    
    # A = r
    # A = x[:,0]
    # A = x[:,1]
    A = x[:,2]
    # pr = plot_DF(A, nmesh = 100)
    pr = plot_all_DF(data, 7, nmesh = 100)
    print(pr)
    print(x0[0], v0[0])