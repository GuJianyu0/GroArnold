#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import analysis_data_distribution as add
import galaxy_models as gm
import fit_rho_fJ as fff



if __name__ == '__main__':

    ##read
    ID = 2 #2,3,8 #10 #5,6,7,9, 4
    # swit = 0
    for swit in np.arange(0,1,1):
        ####SF
        path_work = "~/workroom/"
        path_base = path_work+"0prog/gadget/Gadget-2.0.7/galaxy_general/"
        filename1 = path_base+"debug/ptau_%d_debug.txt"%(ID)
        data1 = np.loadtxt(filename1, dtype=float)

        x1 = data1[:,0:3]
        v1 = data1[:,3:6]
        tau1 = data1[:,6]
        Ints1 = data1[:,8:11]
        ptau1 = data1[:,12]
        ptau_return1 = data1[:,12]

        ####TEPPOD
        filename2 = path_base+"orbit/orbit_particle_%d_time_5.000000e+00.txt"%(ID)
        file_handle = open(filename2, mode="r")
        # file_handle.write("##abcd\n")
        # st = file_handle.readlines()
        # data2_ = copy.deepcopy(a)
        data2_ = [np.array([]), np.array([]), np.array([])]
        st = True
        while st:
            st = file_handle.readline()
            is_read = False
            block = 0
            judge = add.judge_read_line(st)
            if judge in [0, -2, -1, 1]: #not used line
                # print(judge, ": ", st)
                if is_read:
                    is_read = False
                    block += 1
            else: #judge==2; used line
                print(judge, ": ", st)
                if not is_read:
                    is_read = True
                fl = st.split()
                # print("st: ", st)
                # print("fl: ", fl)
                fl = np.float64(fl)
                data2_[swit] = np.append(data2_[swit], fl)
        file_handle.close()



        data2 = data2_[swit]
        print(data2, data2.shape)
        data2 = data2.reshape(-1,21)
        print(data2.shape)
        # exit(0)

        x2 = data2[:, 0:3]
        v2 = data2[:, 3:6]
        time2 = data2[:, 6]
        ABC2 = data2[:, 7:10]
        tau2 = data2[:, 10:13]
        ptau2 = data2[:, 13:16]
        potential2 = data2[:, 16]
        energy2 = data2[:, 17]

        ####plot compare
        pointsize = 10.
        fontsize = 20.
        # plt.subplot(3,1,swit+1)
        alp2 = -ABC2[0,0]
        xplot = tau2[:,swit]
        yplot = ptau2[:,swit]# np.abs(ptau2[:,swit])
        plt.plot([np.min(xplot), np.max(xplot)], [0., 0.], color="black")
        plt.plot([alp2, alp2], [1e-1, 1e1], color="black")
        plt.scatter(xplot, yplot, s=pointsize, 
            color="blue", label=r"$\tau~p_\tau$, TEPPOD")
        xplot = tau1
        yplot = ptau1**0.5*np.sign(ptau1)
        plt.scatter(xplot, yplot, s=pointsize, 
            color="red", label=r"$\tau~p_\tau$, SFFP_SCF")

        # plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"$tau$", fontsize=fontsize)
        plt.ylabel(r"$p_\tau$", fontsize=fontsize)
        plt.title("particleID_%d"%(ID))
        plt.legend(fontsize=fontsize)
        plt.show()






    '''
        ####plot data2
        x = data[:, 0:3]
        v = data[:, 3:6]
        time = data[:, 6]
        ABC = data[:, 7:10]
        tau = data[:, 10:13]
        ptau = data[:, 13:16]
        potential = data[:, 16]
        energy = data[:, 17]

        X = tau[:,swit]
        Y = ptau[:,swit]
        # Y = add.logabs(Y, is_keep_sign=1)
        T = time[:]

        pointsize = 10.
        fontsize = 20.
        plt.subplot(3,1,swit+1)
        plt.scatter(X, Y, s=pointsize, color="red",  label=r"$\tau~p_\tau$")
        plt.scatter(X, T, s=pointsize, color="blue", label=r"$\tau~t$")
        plt.plot([np.min(X), np.max(X)], [0., 0.], color="black")
        # plt.plot([np.min(X), np.max(X)], [1., 1.], color="green")
        plt.xlabel(r"$tau$ and $t$", fontsize=fontsize)
        plt.ylabel(r"$p_\tau$", fontsize=fontsize)
        plt.legend(fontsize=fontsize)
    plt.show()
    '''