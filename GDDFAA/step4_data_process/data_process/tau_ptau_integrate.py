#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sys import argv #swit = int(argv[1])
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel

import analysis_data_distribution as add
import galaxy_models as gm
import fit_rho_fJ as fff
import integrate_methods as im



if __name__ == '__main__':

    ##1: settings
    ID = 2
    # ID = 8

    # path_suffix = "_Small"
    # path_suffix = "_Middle"
    path_suffix = "_Large"

    # index_start = 0
    # index_end = -1
    # index_start = 60
    # index_end = 180
    index_start = 160
    index_end = 1180

    # i_plot = 0
    i_plot = 1

    ##data
    path_work = "~/workroom/"
    path_base = "0prog/gadget/Gadget-2.0.7/galaxy_general/"
    filename = path_work+path_base+"orbit/orbit_particle_%d_endless"%(ID)+path_suffix+".txt"
    data = np.loadtxt(filename, dtype=float)

    alpha = -3.
    beta = -2.
    gamma = -1.

    x = data[index_start:index_end, 0:3]
    v = data[index_start:index_end, 3:6]
    t = data[index_start:index_end, -1]
    r = add.norm_l(x, axis=1)
    tau3 = data[index_start:index_end, 6:9]
    metric_tau3 = data[index_start:index_end, 9:12]
    dot_tau3 = data[index_start:index_end, 15:18]
    p_tau3 = data[index_start:index_end, 21:24]

    for swit in np.arange(3):
        ##process
        tau = tau3[:, swit]
        metric_tau = metric_tau3[:, swit]
        dot_tau = dot_tau3[:, swit]
        p_tau = p_tau3[:, swit]
        J_tau = np.zeros(len(tau))

        # widthKernel = 3.
        # # widthKernel = np.std(p_tau)
        # # p_tau_removeleaping = p_tau
        # gauss_kernel = Gaussian1DKernel(widthKernel)
        # p_tau_smooth_Gauss3 = convolve(p_tau, gauss_kernel)
        # box_kernel = Box1DKernel(widthKernel)
        # p_tau_smooth_Box3 = convolve(p_tau, box_kernel)
        # PP = [p_tau, p_tau_removeleaping, p_tau_smooth_Gauss3, p_tau_smooth_Box3]
    
        index_lims = np.array([])
        t_lims = np.array([])
        tau_lims = np.array([])
        p_tau_lims = np.array([])

        N = len(p_tau)
        for i in np.arange(0, N-1):
            if p_tau[i]*p_tau[i+1]<=0:
                index_lims = np.append(index_lims, i)
                t_lims = np.append(t_lims, t[i])
                tau_lims = np.append(tau_lims, tau[i])
                p_tau_lims = np.append(p_tau_lims, p_tau[i])
        NL = len(p_tau_lims)
        index_lims = index_lims.astype(int)
        J_tau_lims = np.zeros(NL-1)

        ##integrate
        for i in np.arange(0,NL-1):
            idxD = index_lims[i]+1
            idxU = index_lims[i+1]-1 #now use inner left and right, so remove each tail
            tau_itg = tau[idxD:idxU]
            p_tau_itg = p_tau[idxD:idxU]
            J_tau_lims[i] = np.abs(im.integrate_samples_trapezoid_1d(tau_itg, p_tau_itg)*2/np.pi)
            J_tau[idxD:idxU+1] = J_tau_lims[i]
        # print("J_tau: ", J_tau)
        print("J_tau%d: [mean, median, stdandard]: "%(swit), np.mean(J_tau), np.median(J_tau), np.std(J_tau))
        print("t_lims%d: [mean, median, stdandard]: "%(swit), np.mean(t_lims), np.median(t_lims), np.std(t_lims))



        ##plot
        pointsize = 0.2
        fontsize = 10.0
        xplots = [tau, t]
        xplots_name = ["tau", "t"]
        yplots = [tau, t, metric_tau, dot_tau, p_tau, J_tau]
        yplots_name = ["tau", "t", "metric_tau", "dot_tau", "p_tau", "J_tau"]
        N_yplot = len(yplots)
        for j in np.arange(N_yplot):
            plt.subplot(3, N_yplot, swit*N_yplot+j+1)
            plt.plot([np.min(xplots[i_plot]),np.max(xplots[i_plot])], [0.,0.], color="k")
            
            plt.scatter(xplots[i_plot], yplots[j], label=xplots_name[i_plot]+"~"+yplots_name[j], s=pointsize, color=fff.colors[j])
            plt.plot(xplots[i_plot], yplots[j], lw=pointsize/2, color=fff.colors[j])
            plt.grid()
            plt.legend(fontsize=fontsize)

    plt.suptitle("particleID_%d"%(ID))
    plt.show()
    plt.close()





    '''
    ##2: reading
    nameswit = [r"\lambda", r"\tau", r"\nu"]
    for swit in np.arange(3):
        filename = "~/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general%s/snaps/aa/orbit_%d_directintegration_endless.txt" % (scripts[-1], ID)
        # filename = "~/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general/snaps/aa/orbit_%d_directintegration_tau%d.txt" % (ID, swit)
        data = np.loadtxt(filename, dtype=float)
        print(r"length of data: %d", len(data))



        t = np.arange(len(data))*0.01
        x0 = data[:, 0]
        x1 = data[:, 1]
        x2 = data[:, 2]
        tau = data[:, 6+swit]
        metric_tau = data[:, 9+swit]
        dot_tau = data[:, 15+swit]
        p_tau = data[:, 21+swit]
        r = add.norm_l(data[:,0:3], axis=1)
        E = data[:, -1]
        t_is_lims = np.zeros(len(data))
        # add.DEBUG_PRINT_V(1, swit, max(metric_tau), np.log(max(metric_tau)))
        # add.DEBUG_PRINT_V(1, swit, min(metric_tau), np.log(min(metric_tau)))



        ##3: deal with p_tau
        # widthKernel = np.std(p_tau)
        widthKernel = 3.
        # p_tau_removeleaping = p_tau
        gauss_kernel = Gaussian1DKernel(widthKernel)
        p_tau_smooth_Gauss3 = convolve(p_tau, gauss_kernel)
        box_kernel = Box1DKernel(widthKernel)
        p_tau_smooth_Box3 = convolve(p_tau, box_kernel)

        PP = [p_tau, p_tau, p_tau_smooth_Gauss3, p_tau_smooth_Box3]
        TT = list(range(4))
        II = list(range(4))
        TL = list(range(4))
        PL = list(range(4))
        AL = list(range(4))

        for j in np.arange(4):
            ##estimate limits (should by calculating roots)
            # t_lims = np.array([])
            # tau_lims = np.array([])
            # p_tau_lims = np.array([])
            t_lims = np.array(t[0])
            tau_lims = np.array(tau[0])
            p_tau_lims = np.array(PP[0][0])
            pt = PP[j]
            N = len(pt)
            NL = 0
            for i in np.arange(0, N):
                if pt[i]*pt[i-1]<=0:
                    t_is_lims[i] = 1
                    t_lims = np.append(t_lims, t[i])
                    tau_lims = np.append(tau_lims, tau[i])
                    p_tau_lims = np.append(p_tau_lims, pt[i])
                    NL += 1
            t_lims = np.append(t_lims, t[-1])
            tau_lims = np.append(tau_lims, tau[-1])
            p_tau_lims = np.append(p_tau_lims, pt[-1])
            # NL += 1
            # add.DEBUG_PRINT_V(1, t_lims, tau_lims, p_tau_lims, ": lims")

            ##integrate
            T = t_lims[1:NL] #
            I = np.zeros(NL-1) #
            add.DEBUG_PRINT_V(1, NL, len(T), len(I), ": len")
            rmv = np.zeros(0)
            for i in np.arange(0,NL-1): #
                idxD = int(t_lims[i]*100)
                idxU = int(t_lims[i+1]*100)-1
                # limD = tau_lims[i]
                # limU = tau_lims[i+1]
                tau_itg = tau[idxD:idxU]
                p_tau_itg = p_tau[idxD:idxU]
                I[i] = im.integrate_samples_trapezoid_1d(tau_itg, p_tau_itg)*2/np.pi
                if j==1 and idxU-idxD+1<=4:
                    rmv = np.append(rmv, i)
                # add.DEBUG_PRINT_V(1, (np.max(tau_itg)-np.min(tau_itg))*np.max(abs(p_tau_itg))/2, ": itg_estimation")
                # add.DEBUG_PRINT_V(1, tau_itg, p_tau_itg)
                # add.DEBUG_PRINT_V(1, idxD, idxU-idxD+1, I)
            rmv = rmv.astype(int)
            # T = np.delete(T, rmv)
            # I = np.delete(I, rmv)
            print(I)

            TT[j] = T
            II[j] = np.abs(I)
            TL[j] = t_lims
            PL[j] = p_tau_lims
            AL[j] = tau_lims



        ##4: plot limits
        plt.subplot(3,3,(swit)*3+1)
        plt.plot([np.min(t),np.max(t)],[0,0], color="k")
        plt.plot(t, tau, label=r"$\tau$")
        plt.scatter(TL[0], AL[0], label=r"tau at limits when $p_\tau$ original", s=8., color="k")
        # plt.yscale("log")
        plt.legend(fontsize=fontsize/2)

        plt.subplot(3,3,(swit)*3+2)
        plt.plot([np.min(t),np.max(t)],[0,0], color="k")
        plt.plot(t, PP[0], label=r"$p_\tau$ original", color="b")
        plt.plot(t, PP[1], label=r"$p_\tau$ removeTooShortPeriod", color="g")
        plt.plot(t, PP[2], label=r"$p_\tau$ smoothKernelGauss", color="orange")
        plt.plot(t, PP[3], label=r"$p_\tau$ smoothKernelBox", color="r")
        plt.scatter(TL[0], np.zeros(len(TL[0])), label=r"position of limits when $p_\tau$ original", s=8., color="k")
        # plt.yscale("log")
        plt.legend(fontsize=fontsize/2)

        plt.subplot(3,3,(swit)*3+3)
        plt.plot([np.min(t),np.max(t)],[0,0], color="k")
        plt.plot(TT[0], II[0], label=r"$J_\tau$ original", color="b")
        plt.plot(TT[1], II[1], label=r"$J_\tau$ removeTooShortPeriod", color="g")
        plt.plot(TT[2], II[2], label=r"$J_\tau$ smoothKernelGauss", color="orange")
        plt.plot(TT[3], II[3], label=r"$J_\tau$ smoothKernelBox", color="r")
        plt.scatter(TL[0], np.zeros(len(TL[0])), label=r"position of limits when $p_\tau$ original", s=8., color="k")
        # plt.yscale("log")
        plt.legend(fontsize=fontsize/2)



        # ##5: plot energy
        # plt.subplot(5,1,1)
        # plt.scatter(t, E, label=r"$t$~$E$", s=pointsize)
        # plt.plot(t, E, lw=pointsize/2)
        # plt.legend(fontsize=fontsize/2)
        # plt.title("ID_%d"%(ID))

        # plt.subplot(5,1,2)
        # plt.scatter(r, E, label=r"$r$~$E$", s=pointsize)
        # plt.plot(r, E, lw=pointsize/2)
        # plt.legend(fontsize=fontsize/2)
        # plt.ylim(-100000., -70000.)

        # for sw in [0,1,2]:
        #     plt.subplot(5,1,3+sw)
        #     plt.scatter(data[:, 6+sw], E, label=r"$\tau_%d$~$E$" %(sw), s=pointsize)
        #     plt.plot(data[:, 6+sw], E, lw=pointsize/2)
        #     plt.legend(fontsize=fontsize/2)
        #     plt.ylim(-100000., -70000.)

        # whspace = 0.4
        # plt.subplots_adjust(hspace=whspace, wspace=whspace)



        ## the not converge
        # t = t[0:100]
        # x0 = x0[0:100]
        # x1 = x1[0:100]
        # x2 = x2[0:100]
        # tau = tau[0:100]
        # metric_tau = metric_tau[0:100]
        # dot_tau = dot_tau[0:100]
        # p_tau = p_tau[0:100]

    plt.suptitle("particleID_%d"%(ID))
    plt.show()
    plt.close()


    
    #     ##5: plot ellip coor
    #     logbasis = 10.
    #     # metric_tau = add.logP1(metric_tau, logbasis=logbasis) #taust
    #     # dot_tau = add.logP1(dot_tau, logbasis=logbasis)
    #     # p_tau = add.logP1(p_tau, logbasis=logbasis)
    #     # tau = add.logP1(tau, logbasis=logbasis)
    #     metric_tau = add.logabs(metric_tau, logbasis=logbasis, is_keep_sign=0) #log display
    #     # dot_tau = add.logabs(dot_tau, logbasis=logbasis, is_keep_sign=0)
    #     # dot_tau = np.log(dot_tau)/np.log(logbasis)
    #     # p_tau = add.logabs(p_tau, logbasis=logbasis, is_keep_sign=0)
    #     # tau = add.logabs(tau, logbasis=logbasis, is_keep_sign=0)

    #     # var = [x2, metric_tau, dot_tau, p_tau, tau] #[E, t_is_lims]
    #     var = [x2, metric_tau, dot_tau, p_tau, tau, t_is_lims]
    #     lb = [r"rectangular $x$", r"logP1 metric $\log(g_\tau)$", r"logP1 time variation $\dot\tau$", r"logabs momentum $p_\tau$", r"elliptical $\tau$", r"energy", r"t_is_lims"]
    #     cl = ["k", "red", "orange", "green", "blue", "indigo", "purple", "browm"]

    #     for iv in np.arange(len(var)):
    #         plt.subplot(3, len(var), (swit)*len(var)+(iv+1))

    #         plt.xlabel(r"$\tau$", fontsize=fontsize/1.6)
    #         # plt.xlabel(r"t", fontsize=fontsize/1.6)
    #         plt.ylabel(r"%s, $\tau=%s$" % (lb[iv], nameswit[swit]), fontsize=fontsize/1.6)

    #         # plt.plot([-alpha, -alpha], [yl[0], yl[1]], label=r"$\tau=-a^2$", color="orange")
    #         # plt.plot([-beta , -beta ], [yl[0], yl[1]], label=r"$\tau=-b^2$", color="orange")
    #         # plt.plot([-gamma, -gamma], [yl[0], yl[1]], label=r"$\tau=-c^2$", color="orange")
    #         # plt.plot([xl[0], xl[1]], [0., 0.], label=r"the root line, $y=0$", color="k")
    #         # plt.text(-alpha, -yl[0], r"$\tau=-a^2$")
    #         # plt.text(-beta , -yl[0], r"$\tau=-b^2$")
    #         # plt.text(-gamma, -yl[0], r"$\tau=-c^2$")

    #         plt.plot([np.min(t), np.max(t)], [0., 0.], label="line $y=0$ or $log(y)=1$", lw=pointsize)
    #         plt.plot([np.min(t), np.max(t)], [1., 1.], label="line $y=0$ or $log(y)=1$", lw=pointsize)
    #         # plt.plot(x0, x1, color="r", label="xy, with order along orbit", lw=pointsize/6)
    #         # plt.scatter(tau, var[iv], color=cl[iv], label=lb[iv], s=pointsize)
    #         plt.scatter(t, var[iv], color=cl[iv], label=lb[iv], s=pointsize)
    #         # plt.plot(tau, var[iv], color=cl[iv], lw=pointsize/6)
    #         plt.plot(t, var[iv], color=cl[iv], lw=pointsize/6)
    #         # plt.scatter(x0, x1, color="r", label="xy", s=pointsize)

    #         # plt.xlim(1.75, 2.2)
    #         # plt.xlim(0.75, 3.2)
    #         if iv==2:
    #             plt.ylim(-5e1, 5e1)
    #         if iv==3:
    #             # plt.ylim(-1e0, 3e0)
    #             # plt.ylim(-1e2, 3e2)
    #             plt.ylim(-5e3, 5e3)
    #             # plt.ylim(-1e5, 3e6)

    # # plt.title(r"about ellip coordinate along an orbit of particle_ID %d"%(ID)+" \n"+r"(these variables donot have the same units)", fontsize=fontsize)
    # whspace = 0.4
    # plt.subplots_adjust(hspace=whspace, wspace=whspace)
    # plt.legend(fontsize=fontsize/20)
    # fig_tmp = plt.gcf()
    # plt.show()
    # plt.close("all")
    # print("Fig ... Done.")



    ##plot mass E and orb
    t = np.arange(len(data))*0.01
    data_mass = list(range(3))
    x = list(range(3))
    r = list(range(3))
    tau = list(range(3))
    E = list(range(3))

    for level_mass in np.arange(length_model):
        filename = "~/workroom/0prog/gadget/Gadget-2.0.7/galaxy_general%s/snaps/aa/orbit_%d_directintegration_endless.txt" % (scripts[level_mass], ID)
        data = np.array(np.loadtxt(filename, dtype=float))
        data_mass[level_mass] = data
        x[level_mass] = data_mass[level_mass][:, 0:3]
        r[level_mass] = add.norm_l(x[level_mass], axis=1)
        tau[level_mass] = data_mass[level_mass][:, 6:9]
        E[level_mass] = data_mass[level_mass][:, -1]
        E[level_mass] /= np.mean(E[level_mass]) #to display reletive changing

    nameswit = [r"\lambda", r"\mu", r"\nu"]
    # for swit in np.arange(3):

    fig = plt.figure()
    plt.subplot(2,3,1)
    for level_mass in np.arange(length_model):
        plt.scatter(t, E[level_mass], label=r"$t$~$E$ (%s)"%(scripts[level_mass]), s=pointsize)
        plt.plot(t, E[level_mass], lw=pointsize/2)
    plt.legend(fontsize=fontsize)
    # plt.ylim(-100000., -70000.)

    plt.subplot(2,3,2)
    for level_mass in np.arange(length_model):
        plt.scatter(r[level_mass], E[level_mass], label=r"$r$~$E$ (%s)"%(scripts[level_mass]), s=pointsize)
        plt.plot(r[level_mass], E[level_mass], lw=pointsize/2)
    plt.legend(fontsize=fontsize)
    # plt.ylim(-100000., -70000.)

    plt.subplot(2,3,3)

    for sw in [0,1,2]:
        plt.subplot(2,3,sw+1+3)
        for level_mass in np.arange(length_model):
            plt.scatter(tau[level_mass][:,sw], E[level_mass], label=r"$%s$~$E$ (%s)"%(nameswit[sw], scripts[level_mass]), s=pointsize)
            plt.plot(tau[level_mass][:,sw], E[level_mass], lw=pointsize/2)
        plt.legend(fontsize=fontsize)
        # plt.ylim(-100000., -70000.)

    whspace = 0.4
    plt.subplots_adjust(hspace=whspace, wspace=whspace)
    plt.suptitle(r"energy of particle with ID_%d"%(ID))
    plt.show()
    plt.close()



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for level_mass in np.arange(length_model):
        ax.scatter(x[level_mass][:,0], x[level_mass][:,1], x[level_mass][:,2], label=r"orbit (%s)"%(scripts[level_mass]), s=pointsize)
        ax.plot(x[level_mass][:,0], x[level_mass][:,1], x[level_mass][:,2], lw=pointsize/2)
    ax.legend(fontsize=fontsize)
    ax.set_xlabel(r"x", fontsize=fontsize)
    ax.set_ylabel(r"y", fontsize=fontsize)
    ax.set_zlabel(r"z", fontsize=fontsize)

    whspace = 0.4
    plt.subplots_adjust(hspace=whspace, wspace=whspace)
    plt.suptitle(r"orbit of particleID_%d"%(ID))
    plt.show()
    plt.close()
    '''