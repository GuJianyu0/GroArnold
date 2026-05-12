#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def norm_l(a,l=2):
    x=0
    for e in a:
        x += e**l
    return x**(1./l)



if __name__ == '__main__':

    fname = "./data_process/aa/20210314_Bovy13/PF_compare_x00_pl_3.6e5__PM17.txt"
    # fname = "./data_process/aa/20210314_Bovy13/PF_compare_nfw__PM17_x00.txt"
    # fname = "./data_process/aa/20210314_Bovy13/PF_compare_pl2__PM17_x00.txt"
    # fname = "./data_process/aa/20210314_Bovy13/PF_compare_pl2__PM17_Rp0.txt"
    # fname = "./data_process/aa/20210314_Bovy13/PF_compare_pl2_Rp0.txt"
    # fname = "./data_process/aa/20210314_Bovy13/PF_compare.txt"
    data = np.loadtxt(fname)
    xv = data[:, 0:3]
    x0 = xv[:, 0]

    P_ds = data[:, 6]
    P_gg = data[:, 7]
    P_rate = data[:, 8]

    Fx_ds = data[:, 9]
    Fx_gg = data[:, 10]
    Fx_rate = data[:, 11]

    F_ds = data[:, 18]
    F_gg = data[:, 19]
    F_rate = data[:, 20]

    Pxx_ds = data[:, 21]
    Pxx_gg = data[:, 22]
    Pxx_rate = data[:, 23]

    l = len(P_ds)
    n = np.arange(l)
    r = np.zeros(l)



    ##plot x00-P
    plt.scatter(x0, P_ds, s=0.8, label="D potential")
    plt.scatter(x0, P_gg, s=0.8, label="F potential")
    plt.plot([0., max(x0)], [0., 0.], color="black")
    plt.xlabel(r'$x\quad \mathrm{kpc}$', fontsize=20)
    plt.ylabel(r'potential on x-axis {x, 0., 0.}', fontsize=20)
    plt.legend()
    plt.savefig(fname+"_P.png", dpi=200)
    plt.show()

    ##plot x00-Fx
    plt.scatter(x0, Fx_ds, s=0.8, label="D potential")
    plt.scatter(x0, Fx_gg, s=0.8, label="F potential")
    plt.plot([0., max(x0)], [0., 0.], color="black")
    plt.xlabel(r'$x\quad \mathrm{kpc}$', fontsize=20)
    plt.ylabel(r'Fx on x-axis {x, 0., 0.}', fontsize=20)
    plt.legend()
    plt.savefig(fname+"_Fx.png", dpi=200)
    plt.show()

    ##plot x00-Pxx
    plt.scatter(x0, Pxx_ds, s=0.8, label="D potential")
    plt.scatter(x0, Pxx_gg, s=0.8, label="F potential")
    plt.plot([0., max(x0)], [0., 0.], color="black")
    plt.xlabel(r'$x\quad \mathrm{kpc}$', fontsize=20)
    plt.ylabel(r'P_derive_xx on x-axis {x, 0., 0.}', fontsize=20)
    plt.legend()
    plt.savefig(fname+"_Pxx.png", dpi=200)
    plt.show()



    # ##plot xy-P
    # fig = plt.figure(figsize=(16, 12), dpi=80, facecolor=(0.0, 0.0, 0.0))
    # ax = Axes3D(fig)
    # ax.grid(True)
    # lim = 2*np.median(r)
    # ax.set_xlim(-lim, lim)
    # ax.set_ylim(-lim, lim)
    # ax.scatter(xv[:,0], xv[:,1], P_ds, color="blue", s=0.8)
    # ax.scatter(xv[:,0], xv[:,1], P_gg, color="red", s=0.8)
    # ax.plot3D([-lim,lim],[0,0],[0,0], color="red", linewidth=0.5)
    # ax.plot3D([0,0],[-lim,lim],[0,0], color="red", linewidth=0.5)
    # ax.plot3D([0,0],[0,0],[-lim,lim], color="red", linewidth=0.5)
    # ax.set_xlabel(r'$x\quad \mathrm{kpc}$', fontsize=20)
    # ax.set_ylabel(r'$y\quad \mathrm{kpc}$', fontsize=20)
    # ax.set_zlabel(r'potential', fontsize=20)
    # plt.savefig(fname+".png", dpi=100)
    # plt.show()



    # ##check PF
    # F_ds = data[:, 9::3]
    # F_gg = data[:, 10::3]
    # F_rate = data[:, 11::3]
    # f_ds = np.zeros(l)
    # f_gg = np.zeros(l)
    # f_rate = np.zeros(l)
    # for i in range(l):
    #     r[i] = norm_l(xv[i])
    #     f_ds[i] = norm_l(F_ds[i])
    #     f_gg[i] = norm_l(F_gg[i])
    # f_rate = (f_ds-f_gg)/f_gg

    # for k in np.arange(1,7):
    #     plt.subplot(3,2,k)
    #     if(k==1):
    #         plt.scatter(r, P_ds, s=0.1)
    #         plt.scatter(r, P_gg, s=0.1)
    #         plt.ylabel(r"potential compare $(km/s)^2$", fontsize=10)
    #     if(k==2):
    #         plt.scatter(r, P_rate, s=0.1)
    #         plt.ylabel(r"potential rate, (ds-gg)/gg", fontsize=10)

    #     if(k==3):
    #         plt.scatter(r, Fx_ds, s=0.1)
    #         plt.scatter(r, Fx_gg, s=0.1)
    #         plt.ylabel(r"Fx compare $N$", fontsize=10)
    #     if(k==4):
    #         plt.scatter(r, Fx_rate, s=0.1)
    #         plt.ylabel(r"Fx rate, (ds-gg)/gg", fontsize=10)

    #     if(k==5):
    #         plt.scatter(r, f_ds, s=0.1)
    #         plt.scatter(r, f_gg, s=0.1)
    #         plt.ylabel(r"F_norm compare $N$", fontsize=10)
    #     if(k==6):
    #         plt.scatter(r, f_rate, s=0.1)
    #         plt.ylabel(r"F_norm rate, (ds-gg)/gg", fontsize=10)
    #     plt.xlabel(r"radical distance $kpc$", fontsize=10)
    # plt.show()

    # ##no use
    # # dataall = [P_ds, P_gg, P_rate, 
    # #         Fx_ds, Fx_gg, Fx_rate, 
    # #         F_ds, F_gg, F_rate]
    # # enum_ds = np.array([0, 0, 0, 3, 0, 6])
    # # enum_gg = np.array([0, 1, 0, 4, 0, 7])
    # # enum_rate = np.array([0, 0, 2, 0, 5, 0, 8])
    # # for k in np.arange(1,6):
    # #     plt.subplot(3,2,k)
    # #     if(k%2): #k = 1,3,5
    # #         plt.scatter(r, dataall[enum_ds[k]], s=0.1)
    # #         plt.scatter(r, dataall[enum_gg[k]], s=0.1)
    # #         plt.xlabel(r"1", fontsize=10)
    # #         plt.ylabel(r"2", fontsize=10)
    # #     else: #k = 2,4,6
    # #         plt.scatter(r, dataall[enum_rate[k]], s=0.1)
    # #         plt.xlabel(r"1", fontsize=10)
    # #         plt.ylabel(r"2", fontsize=10)
    # #     plt.legend()
    # # plt.show()