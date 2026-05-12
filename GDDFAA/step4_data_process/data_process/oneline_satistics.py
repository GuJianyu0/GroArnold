#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import tables
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import read_datas

def plot_potential_compare():
    filename1="/home/darkgaia/0prog/Category+/pieces/kdtree2/interp_knn_line.txt"
    # filename1="/home/darkgaia/0prog/Category+/pieces/kdtree2/interp_knn_allall_err.txt"
    # filename1="/home/darkgaia/0prog/Category+/pieces/kdtree2/get_available_no_knn_pot_all.txt"
    # filename1="/home/darkgaia/0prog/Category+/pieces/kdtree2/get_available_no_knn_pot.txt"
    filename2="/home/darkgaia/0prog/Category+/pieces/kdtree2/interp_knn_line_avi.txt"
    filename11="/home/darkgaia/0prog/Category+/pieces/kdtree2/interp_knn_line_Gauss.txt"
    filename12="/home/darkgaia/0prog/Category+/pieces/kdtree2/interp_knn_line_MQ.txt"
    filename13="/home/darkgaia/0prog/Category+/pieces/kdtree2/interp_knn_line_IMQ.txt"
    filename14="/home/darkgaia/0prog/Category+/pieces/kdtree2/interp_knn_line_TP.txt"

    A1 = np.loadtxt(filename1, dtype=float)
    A2 = np.loadtxt(filename2, dtype=float)
    B11 = np.loadtxt(filename11, dtype=float)
    B12 = np.loadtxt(filename12, dtype=float)
    B13 = np.loadtxt(filename13, dtype=float)
    B14 = np.loadtxt(filename14, dtype=float)

    x  = A1[:,0]
    y0=A1[0,1]; z0=A1[0,2]
    f1 = A1[:,3] #interp_0
    f3 = A1[:,4] #adding
    # f4 = A1[:,5] #gadget data
    # f2 = A2[:,3] #interp_screen
    x  = B11[:,0]
    f10 = B11[:,4] #adding
    f11 = B11[:,3]
    f12 = B12[:,3]
    f13 = B13[:,3]
    f14 = B14[:,3]

    rate = f1/f3
    print(np.mean(rate))
    print(min(f1)/min(f3))

    plt.figure()
    plt.plot(x, f10, color="black", label="directly adding")
    plt.plot(x, f11, label="interpolation_RBF_Gauss")
    plt.plot(x, f12, label="interpolation_RBF_MQ")
    plt.plot(x, f13, label="interpolation_RBF_IMQ")
    plt.plot(x, f14, label="interpolation_RBF_TP")

    # plt.scatter(x, f1, s=1., label="interpolation_RBF without inner screening")
    # plt.scatter(x, f2, s=1., label="interpolation_RBF with inner screening")
    # plt.scatter(x, f3, s=1., label="directly adding")
    # plt.scatter(x, f4, s=1., label="gadget data")
    # plt.scatter(x, rate, s=1., label="gadget data")

    plt.grid()
    plt.legend(fontsize=10)
    plt.title("potential-x comparation:\nposition = {(x, 1., 1.)}", fontsize=20)
    # plt.title("potential-x comparation:\nposition = {(xi+1, yi+1., zi+1.)}", fontsize=20)
    # plt.title("potential-x comparation:\nposition = {(x, 1., 1.)}", fontsize=20)
    # plt.title("potential-x distribution:\nposition = (x, %f, %f)" % (y0, z0), fontsize=20)
    # plt.savefig("potentile test for adding and interpolation_RBF")
    plt.show()

def plot_neareast_distribution():
    filename1 = "/home/darkgaia/0prog/gadget/gadget-2.0.7/data_process/snap20.txt"
    data = np.loadtxt(filename1)
    filename2 = "/home/darkgaia/0prog/Category+/pieces/kdtree2/oneline_statistics_nn.txt"
    idx = np.loadtxt(filename2, dtype=int)
    filename3 = "/home/darkgaia/0prog/Category+/pieces/kdtree2/oneline_statistics_potentialtest.txt"
    pot = np.loadtxt(filename3, dtype=float)

    l=478 #0,1,...,999
    idx1 = idx[l,:] #knn idx the first particle
    x0=-200.+400./1000.*l; y0=1; z0=1; pot0=pot[l,0] #tgt point
    print(pot0)

    fig2 = plt.figure(figsize=(16, 12), dpi=80, facecolor=(0.0, 0.0, 0.0))
    ax = Axes3D(fig2)
    ##tgt data
    ax.scatter(x0, y0, pot0, color="red", s=2., label="tgt pot")
    #ax.plot(np.ones(100)*x0, np.ones(100)*y0, np.linspace(0, pot0, 100), color="red")
    ##knn data
    ax.scatter(data[idx1,0], data[idx1,1], data[idx1,7], color="black", s=2., label="knn pot")
    # for i in idx1:
    #     x1 = np.ones(100)*data[i,0]
    #     y1 = np.ones(100)*data[i,1]
    #     z1 = np.linspace(0, data[i,7], 100)
    #     ax.plot(x1, y1, z1, color="black")
    ##all data
    ax.scatter(data[:,0], data[:,1], data[:,7], color="blue", s=0.1, label="all pot")
    ##xOy plane
    # X = np.arange(-abs(np.sqrt(x0**2+y0**2)), abs(np.sqrt(x0**2+y0**2)), 0.2)
    # Y = np.arange(-abs(y0), abs(y0), 0.2)
    # X, Y = np.meshgrid(X, Y)
    # Z = 0*X+0*Y+0
    # ax.plot_surface(X, Y, Z, linewidth=0.5)

    ax.grid(True)
    ax.set_xlabel(r'$x coor,\quad kpc$', fontsize=15)
    ax.set_ylabel(r'$y coor,\quad kpc$', fontsize=15)
    ax.set_zlabel(r'$potential,\quad G*M_{sun}/kpc$', fontsize=15)
    ax.text3D(0,0,0, r'O', fontsize=25)
    ax.set_title("potential-xy distribution:\ntarget position = \n (%f, %f, %f)" % (x0, y0, z0), fontsize=20)
    ax.legend(fontsize=15)
    plt.show()

if __name__ == '__main__':

    plot_potential_compare()


####discard
    # plt.plot(x, f1, label="interpolation_RBF without inner screening")
    # # plt.plot(x, f2, label="interpolation_RBF with inner screening")
    # plt.plot(x, f3, label="directly adding")
    # plt.plot(x, f4, label="gadget data")
    # # plt.plot(x, rate, label="potential_interpolation_noscreening / potential_adding")