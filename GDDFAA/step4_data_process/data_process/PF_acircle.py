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

    fname = "./data_process/aa/20210419_NFWBovy13/acircle.PF"
    data = np.loadtxt(fname)
    x = data[:, 0:3]
    v = data[:, 3:6]
    P_D = data[:, 6]
    P_F = data[:, 7]

    l = len(x[:,0])
    n = np.arange(l)
    r = np.zeros(l)
    v = np.zeros(l)
    P_M = np.ones(l)*np.mean(P_D)

    fig = plt.figure(figsize=(16, 12), dpi=80, facecolor=(0.0, 0.0, 0.0))
    ax = Axes3D(fig)
    ax.grid(True) # ax.set_axis_off() #remove all relevent axis
    ax.scatter(x[:,0], x[:,1], P_D[:], s=2., color="blue", label="Data potential")
    ax.plot3D(x[:,0], x[:,1], P_M[:], linewidth=1., color="red", label="the average")
    # ax.set_xlim(min(ps_rr), max(ps_rr))
    # ax.set_ylim(min(ps_vv), max(ps_vv))
    # ax.set_zlim(0.5, 2.)
    # lim = max(x[:,0])
    # ax.plot3D([-lim,lim],[0,0],[0,0], color="red", linewidth=0.5)
    # ax.plot3D([0,0],[-lim,lim],[0,0], color="red", linewidth=0.5)
    # ax.plot3D([0,0],[0,0],[-lim,lim], color="red", linewidth=0.5)
    ax.plot3D([0,0],[0,0],[min(P_D),max(P_D)], color="black", linewidth=1.)
    ax.text3D(-1.,-1.,-1., r'O', fontsize=20)
    ax.set_xlabel(r'$x\quad \mathrm{kpc}$', fontsize=20)
    ax.set_ylabel(r'$y\quad \mathrm{kpc}$', fontsize=20)
    ax.set_zlabel(r'$P\quad \mathrm{km/s}^2$', fontsize=20)
    ax.legend()
    ax.view_init(elev = 10., azim = 330.)
    plt.legend()
    plt.savefig(fname+".png")
    plt.show()