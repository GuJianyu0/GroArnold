#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import read_a_data

def plot_compare_interp():
    # filename1 = "./20201109_forcesunits_0115.txt"
    # filename1 = "./20201109_forcesunits_soft1.txt"
    # filename1 = "./20201109_forcesunits_soft005.txt"
    filename1 = "./20201118_pfunits.txt"
    A1 = np.loadtxt(filename1, dtype=float)

    p0 = A1[:,0] #formula
    p1 = A1[:,1] #data interp
    p2 = A1[:,2] #simply add

    f0 = A1[:,3:6]  #formula
    f1 = A1[:,6:9]  #data interp
    f2 = A1[:,9:12] #simply add
    fx0 = f0[:,0]
    fx1 = f1[:,0]
    fx2 = f2[:,0]

    x = A1[:,12]
    y = A1[:,13]
    z = A1[:,14]

    opt = 1
    if opt == 1:
        p0 = np.log(abs(p0))
        p1 = np.log(abs(p1))
        p2 = np.log(abs(p2))
        r = np.sqrt(x**2+y**2+z**2)
        x = np.log(abs(r))

    meanratep = [np.mean(p1/p0), np.mean(p2/p0), np.mean(p2/p1)]
    meanratep_total = [np.mean(p1)/np.mean(p0), np.mean(p2)/np.mean(p0), np.mean(p2)/np.mean(p1)]
    p1 /= meanratep[0]
    p2 /= meanratep[1]
    meanratefx = [np.mean(fx1/fx0), np.mean(fx2/fx0), np.mean(fx2/fx1)] # /+mean is not mean+/
    meanratefx_total = [np.mean(fx1)/np.mean(fx0), np.mean(fx2)/np.mean(fx0), np.mean(fx2)/np.mean(fx1)] #??
    f1[:,0] /= meanratefx[0]
    f2[:,0] /= meanratefx[1]
    # print(meanratep, meanratep_total, meanratefx, meanratefx_total)
    print(meanratep, meanratefx)

    plt.figure()
    if opt == 1:
        plt.axes(yscale = "log")
        plt.axes(xscale = "log")

    plt.scatter(x, p1, color="blue" , s=2., label="potential interp")
    plt.scatter(x, p2, color="red"  , s=2., label="potential simply add")
    plt.scatter(x, p0, color="black", s=2., label="potential formula")

    # plt.scatter(x, fx0, color="black", s=5., label="forces formula")
    # plt.scatter(x, fx1, color="blue" , s=5., label="forces interp")
    # plt.scatter(x, fx2, color="red"  , s=5., label="forces simply add")

    #plt.grid()
    plt.legend(fontsize=10)
    plt.title("compare: position = {data_nodes}", fontsize=20)
    # plt.title("compare_interp forces:\nposition = {(x, 1., 1.)}", fontsize=20)
    plt.show()
    return 0

if __name__ == '__main__':

    plot_compare_interp()

####discard