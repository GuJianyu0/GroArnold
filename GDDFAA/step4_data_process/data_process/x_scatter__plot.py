#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def norm_l(a,l=2):
    x=0
    for e in a:
        x += e**l
    return x**(1./l)

def n_xyz__plot(ff,ii):
    filename = ff+"/snaps/snapshot_%03d.txt" % ii
    snap = str(ii)
    # filename = ff+"/txt/snap_"+snap
    # filename += ".halo.txt" #txt
    # filename += ".disk.txt" #txt
    # filename += ".gasdisk.txt" #txt
    # filename += ".stellarbulge.txt" #txt
    A = np.loadtxt(filename, dtype=float)
    #select #gas 0~4000, halo 4000~24000, disk 24000~34000, bulge 34000~36000
    xyz = A[24000:34000, 0:3]
    l = len(xyz)
    r = np.zeros(l)
    for i in range(l):
        r[i] = norm_l(xyz[i])
    rm = np.median(r)*2

    fig = plt.figure(figsize=(16, 12), dpi=80, facecolor=(0.0, 0.0, 0.0))
    ax = Axes3D(fig)
    ax.grid(True)
    # ax.set_axis_off() #remove all relevent axis
    lim = rm #10. #about scale length
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], color="blue", s=0.5)
    # ax.arrow(-lim,0, 2*lim,0) #only 2d??
    ax.plot3D([-lim,lim],[0,0],[0,0], color="red", linewidth=0.5)
    ax.plot3D([0,0],[-lim,lim],[0,0], color="red", linewidth=0.5)
    ax.plot3D([0,0],[0,0],[-lim,lim], color="red", linewidth=0.5)
    ax.set_xlabel(r'$x\quad \mathrm{kpc}$', fontsize=20)
    ax.set_ylabel(r'$y\quad \mathrm{kpc}$', fontsize=20)
    ax.set_zlabel(r'$z\quad \mathrm{kpc}$', fontsize=20)
    # ax.text3D(-10.,-10.,-10., r'O', fontsize=20)
    # ax.legend("space distribution")
    plt.savefig(ff+"/pics/snapshot_"+snap+".png", dpi=200)
    print("Fig "+snap+" done.")
    # plt.show() #
    plt.close("all")

if __name__ == '__main__':

    ff = "/home/darkgaia/0prog/gadget/gadget-2.0.7/galaxy_Bovy13"
    # ff = "/home/darkgaia/0prog/gadget/gadget-2.0.7/galaxy_PM17"
    # ff = sys.argv[1]
    n_10 = 1
    n_1  = 10
    for j in range(n_10):
        for i in range(0+j*10,j*10 +n_1):
            print(i," ")
            n_xyz__plot(ff,i)