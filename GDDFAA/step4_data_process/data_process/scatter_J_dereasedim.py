#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import pdb
from tqdm import tqdm

import analysis_data_distribution as add
import galaxy_models as gm
import fit_rho_fJ as fff

def potential_compare_rotate(circle_rigid, tx, *p):

    r = circle_rigid[0]
    pos = np.array([0.,0.,0.]) #circle_rigid[1]
    angle = np.array([0.,0.,0.]) #circle_rigid[2]
    rs = p[0]
    softening = 0.05
    n = p[1]
    l = len(tx)
    nP = 0

    tp = np.zeros((l,n))
    for i in np.arange(l):
        x = tx[i][:,0:3]
        x = x-np.mean(x,axis=0)
        add.DEBUG_PRINT_V(1, x[0])
        nP = len(x)
        x_target = np.zeros(x.shape) #x
        for j in np.arange(n):
            phi = float(j)*2*np.pi/n
            x_target[:,0] = r*np.cos(phi)
            x_target[:,1] = r*np.sin(phi)
            x_target[:,2] = 0.
            # dr = ( (x_target[:,0]-x[:,0])**2 +(x_target[:,1]-x[:,1])**2 +(x_target[:,2]-x[:,2])**2 )**0.5
            dr = add.norm_l(x, x_target, axis=1)
            dr = (dr**2 + softening**2)**0.5
            tp[i,j] = -np.sum(1./dr) #pot = G*sum(mi/ri)
        print("potential_compare_rotate(): tp[%d]" % (i))

    return tp/nP*rs

def plot_surf(tp, t0=0., kt=0.01, is_show=1):

    nt, nphi = tp.shape
    t = np.arange(nt)*kt+t0
    phi = np.arange(nphi)*2*np.pi/nphi
    X, Y = np.meshgrid(phi, t)
    # Z = np.sqrt(X**2+Y**2)
    Z = tp

    fig = plt.figure(dpi=300)
    # fig = plt.figure(dpi=300, figsize=(10,2)) #show, not axis length
    pointsize = 0.2
    fontsize = 6.0
    dpi = 500
    ax=fig.add_subplot(111,projection='3d') # ax = Axes3D(fig)
    ax.grid(True) # ax.set_axis_off()
    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.6, 1., 0.6, 1.]))
    # klim = 0.9
    # lim=200.
    # ax.set_xlim(-lim, lim)
    # ax.set_ylim(-lim, lim)
    # ax.set_zlim(-lim, lim)
    # # ax.arrow(-lim,0, 2*lim,0) #only 2d??
    # ax.plot3D([-lim,lim],[0,0],[0,0], color="red", linewidth=pointsize)
    # ax.plot3D([0,0],[-lim,lim],[0,0], color="red", linewidth=pointsize)
    # ax.plot3D([0,0],[0,0],[-lim,lim], color="red", linewidth=pointsize)

    # ax.scatter(x[:,0], x[:,1], x[:,2], color="blue", s=pointsize)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
    position=fig.add_axes([0.1, 0.3, 0.02, 0.5]) #x pos, ypos, width, shrink
    fig.colorbar(surf, cax=position, aspect=5)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # ax.legend(fontsize=fontsize, loc=0)

    ax.set_xlabel(r"angle in the circle $(2\pi/2\pi)$", fontsize=fontsize)
    ax.set_ylabel(r"time $(Gyr)$", fontsize=fontsize)
    ax.set_zlabel(r"potential $(GM/r_s)$", fontsize=fontsize)
    ax.set_title(r"potential varying in a circle with $radius=r_s$", fontsize=fontsize)
    # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)

    fig_tmp = plt.gcf()
    fig_tmp.savefig("savefig/potential_period_circle.png", format='png', dpi=dpi, bbox_inches='tight')
    if is_show == 1:
        plt.show()
    print("Fig ... Done.")
    plt.close("all")

def plot_scatter2d_compare_formula_and_data(l1, l2, is_show=1):

    x1 = l1[0]
    y1 = l1[1]
    c1 = l1[2]
    x2 = l2[0]
    y2 = l2[1]
    c2 = l2[2]

    dpi = 300
    fig = plt.figure(dpi=dpi)
    pointsize = 0.2
    fontsize = 6.0
    ax=fig.add_subplot(111) # ax = Axes3D(fig)
    ax.grid(True) # ax.set_axis_off()
    ax.set_xlabel(r"x-axis", fontsize=fontsize)
    ax.set_ylabel(r"y-axis", fontsize=fontsize)

    ax.scatter(x1, y1, s=pointsize*2, label=c1)
    ax.scatter(x2, y2, s=pointsize*0.5, label=c2)
    plt.legend(fontsize=fontsize)

    fig_tmp = plt.gcf()
    fig_tmp.savefig("savefig/scatter2d_dstpot.png", format='png', dpi=dpi, bbox_inches='tight')
    if is_show == 1:
        plt.show()
    print("Fig ... Done.")
    plt.close("all")

def plot_scatter2d_rate(x, k_median=8., is_show=1):

    dpi = 300
    fig = plt.figure(dpi=dpi)
    pointsize = 0.2
    fontsize = 6.0
    ax=fig.add_subplot(111) # ax = Axes3D(fig)
    ax.grid(True) # ax.set_axis_off()

    ax.scatter(x[:,0], x[:,1], color="blue", s=pointsize)

    ax.set_xlabel(r"x-axis", fontsize=fontsize)
    ax.set_ylabel(r"y-axis", fontsize=fontsize)

    if k_median>1e-10:
        # klim = 0.9
        # lim=200.
        # ax.set_xlim(-lim, lim)
        # ax.set_ylim(-lim, lim)
        # ax.set_zlim(-lim, lim)
        klim = k_median
        lim = np.median(x,axis=0)*klim
        minx = np.min(x,axis=0)
        ax.set_ylim(-8., 8.)
        # ax.set_ylim(-1., 1.)
        # ax.set_xlim(-lim[0]*0., lim[0])
        # ax.set_ylim(-lim[1]*0., lim[1])
        # ax.set_xlim(minx[0], lim[0])
        # ax.set_ylim(minx[1], lim[1])
        # ax.set_xscale("log")
        # ax.set_yscale("log")

    fig_tmp = plt.gcf()
    fig_tmp.savefig("savefig/scatter2d_dstpot.png", format='png', dpi=dpi, bbox_inches='tight')
    if is_show == 1:
        plt.show()
    print("Fig ... Done.")
    plt.close("all")

def plot_scatter3d_dd(x, k_median=0, is_show=1):

    dpi = 300
    fig = plt.figure(dpi=dpi)
    pointsize = 0.2
    fontsize = 6.0
    ax=fig.add_subplot(111,projection='3d') # ax = Axes3D(fig)
    ax.grid(True) # ax.set_axis_off()

    ax.scatter(x[:,0], x[:,1], x[:,2], color="blue", s=pointsize)
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
    # position=fig.add_axes([0.1, 0.3, 0.02, 0.5]) #x pos, ypos, width, shrink
    # fig.colorbar(surf, cax=position, aspect=5)
    # # ax.arrow(-lim,0, 2*lim,0) #only 2d??
    # ax.plot3D([-lim,lim],[0,0],[0,0], color="red", linewidth=pointsize)
    # ax.plot3D([0,0],[-lim,lim],[0,0], color="red", linewidth=pointsize)
    # ax.plot3D([0,0],[0,0],[-lim,lim], color="red", linewidth=pointsize)
    # ax.legend(fontsize=fontsize, loc=0)

    ax.set_xlabel(r"x-axis", fontsize=fontsize)
    ax.set_ylabel(r"y-axis", fontsize=fontsize)
    ax.set_zlabel(r"z-axis", fontsize=fontsize)
    # ax.set_title(r"action scatter", fontsize=fontsize)
    # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)
    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.6, 1., 0.6, 1.]))

    if k_median>0:
        # klim = 0.9
        # lim=200.
        # ax.set_xlim(-lim, lim)
        # ax.set_ylim(-lim, lim)
        # ax.set_zlim(-lim, lim)
        klim = k_median
        lim = np.median(x,axis=0)*klim
        minx = np.min(x,axis=0)
        ax.set_zlim(-8., 8.)
        # ax.set_xlim(-lim[0]*0., lim[0])
        # ax.set_ylim(-lim[1]*0., lim[1])
        # ax.set_zlim(-lim[2]*0., lim[2])
        # ax.set_xlim(minx[0], lim[0])
        # ax.set_ylim(minx[1], lim[1])
        # ax.set_zlim(minx[2], lim[2])
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        # ax.set_zscale("log")

    fig_tmp = plt.gcf()
    fig_tmp.savefig("savefig/scatter3d_dstpot.png", format='png', dpi=dpi, bbox_inches='tight')
    if is_show == 1:
        plt.show()
    print("Fig ... Done.")
    plt.close("all")

if __name__ == "__main__":

    ## name
    gmn = ""
    # gmn = "_1_NFW"
    # gmn = "_11_NFW_triaxial"
    # gmn = "_41_EinastoUsual"
    name = "rho_PL2" # special
    
    ## model settings
    M = 137.
    ls = 19.6
    ds = 0.004568 #0.000891
    Js = (gm.G*M*ls)**0.5
    id_relative_compare = 1
    MG = fff.Model_galaxy(M, ls, ds)
    # MG.set_value("power1",          np.array([1., 1., 1e-3, 2.e0]))
    # MG.set_value("power2",          np.array([3., 3., 2.e0, 1e1]))
    MG.set_value("power1",          np.array([1., 1., 1e-4, 1e1]))
    MG.set_value("power2",          np.array([3., 3., 2.e0, 1e2]))
    MG.set_value("density_scale",   np.array([ds, ds, ds*1e-1, ds*1e1]))
    MG.set_value("length_scale",    np.array([ls, ls, ls*1e-1, ls*1e1]))
    MG.set_value("axis_ratio_y",    np.array([1., 1., 0.1, 10.]))
    MG.set_value("axis_ratio_z",    np.array([1., 1., 0.1, 10.]))
    MG.set_value("rotate_angle_x",  np.array([0., 2*np.pi, 0., 2*np.pi])) # np.pi #divide by zero
    MG.set_value("rotate_angle_y",  np.array([0., 2*np.pi, 0., 2*np.pi]))
    MG.set_value("rotate_angle_z",  np.array([0., 2*np.pi, 0., 2*np.pi]))
    MG.set_value("log_penalty",     np.array([-10., -10., -100., 1.]))
    # MG.set_value("coef_total",      np.array([1., 1., 0.5, 2.]))
    MG.set_value("action_scale",    np.array([Js, Js, Js*1e-1, Js*1e1]))

    ## func
    f = gm.rho_doublepowerlaw_triaxial_log # special
    params_name = ["power1", "power2", "density_scale", "length_scale", 
                "axis_ratio_y", "axis_ratio_z", 
                "rotate_angle_x", "rotate_angle_y", "rotate_angle_z", 
                "log_penalty"] # special
    nf = len(params_name)
    WF = fff.Wrap_func(MG, f, nf)
    WF.set_bounds(params_name)
    fs = WF.funcfit

    ## read
    galaxymodel_name = "galaxy_general"+gmn+"/"
    RD = fff.Read_data_galaxy(MG, gmn=galaxymodel_name)
    
    # snapshot_begin  = 000
    snapshot_begin  = 500
    snapshot_end    = 501
    N_snapshot      = 1
    kt = 0.01*(snapshot_end-snapshot_begin)/N_snapshot
    t0 = 0.01*snapshot_begin
    snapshot_sequence = np.linspace(snapshot_begin, snapshot_end, N_snapshot+1)[0:-1].astype(int)
    tx = list(range(N_snapshot))
    for i in snapshot_sequence: #snapshot_sequence[i]
        bd = 1e5 #4e4 #1e5 #1e6
        # dos = RD.data_original_simulationoOutput(i)
        # dos = RD.data_original_snapshot(i)
        # x, y = RD.data_sample_screen(dos, x_down=0.0002, x_up=200000., is_logy=True, col_x=9, col_y=0)
        # dos = RD.data_original_AAOutput(i, "combine")
        dos = RD.data_original_AAOutput(i, "all")
        # dos = RD.data_original_AAOutput(i, "directorbit")
        # print("data median: ", np.median(dos, axis=0))
        # x = dos[:,12:15]
        # x = dos[:,21:24]
        # x = dos[:,30:33] #formula trixial Fudge
        # x = dos[:,45:48]
        # x = dos[:,-24:-21]
        x = dos[:,-15:-12] #data trixial Fudge #all dos[45]===0, and others wrong: jr, JR, Jl and Jn
        # dos_pp = RD.data_original_AAOutput_pseudoPeriod(i)
        # x_pp = dos_pp[:, 12:15]
        # x_pp = dos[:, 78:81]
        x_pp = dos[:, 82:85]

        # x[:, :] = x_pp[:, :]
        # x[:, 0] = x_pp[:, 0]

        x = abs(x)
        x_coor = dos[:,0:3]
        v_coor = dos[:,3:6]
        Phi_FD = dos[:,[10,11]]
        Jlam_FD = dos[:,[30,-15]]
        orbitStackeltype_FD = dos[:,[34,-11]]

        x, cl, cnl = add.screen_boundary(x, 1./bd, bd)
        # x, cl, cnl = add.screen_boundary(np.log(x), -20, -20.5)
        x_coor = x_coor[cl]
        v_coor = v_coor[cl]
        Phi_FD = Phi_FD[cl]
        Jlam_FD = Jlam_FD[cl]
        r_init = add.norm_l(x_coor, 0, axis=1, l=2)
        V_init = add.norm_l(v_coor, 0, axis=1, l=2)
        dlt_Phi = add.relative_error(Phi_FD[:,1], Phi_FD[:,0])
        # plt.scatter(r_init, dlt_Phi)
        # plt.show()
        # wp = np.where(dlt_Phi>=0.)[0]
        # add.DEBUG_PRINT_V(0, len(wp))
        # # dlt_Phi = dlt_Phi+1
        # dlt_Phi = abs(dlt_Phi)
        # dlt_Phi = np.log(dlt_Phi)
        dlt_Jlam = add.relative_error(Jlam_FD[:,1], Jlam_FD[:,0])
        # dlt_Jlam = dlt_Jlam+1
        # # dlt_Jlam = abs(dlt_Jlam)
        # dlt_Jlam = np.log(dlt_Jlam)
        orbitStackeltype_FD = orbitStackeltype_FD[cl]
        
        # Jlam = x[:,0]
        # wless = np.where(Jlam<1./bd)[0]
        # wgreater = np.where(Jlam>bd)[0]
        # dosg = dos[wgreater]
        # rg = ( dosg[:,0]**2 +dosg[:,1]**2 +dosg[:,2]**2 )**0.5
        # add.DEBUG_PRINT_V(1, wgreater)
        # add.DEBUG_PRINT_V(1, len(wless), len(wgreater))
        # add.DEBUG_PRINT_V(1, np.median(rg), np.mean(rg))
        # # sum(np.array([ -3.34004e+01, -1.70423e+01, 3.04515e+01 ])**2)**0.5
        # add.DEBUG_PRINT_V(0, len(Phi_FD))
        # lJl = len( np.where(x[:,0]<np.median(x[:,0])*2.)[0] )
        # lJl = len( np.where(x[:,0]<4.e4)[0] )
        # add.DEBUG_PRINT_V(1, lJl)
        wg = np.where(dlt_Jlam>7.)[0]+1
        add.DEBUG_PRINT_V(1, wg, len(wg))

        PFl = [r_init, np.log(-Phi_FD[:,0]), "by formula potential"]
        PDl = [r_init, np.log(-Phi_FD[:,1]), "by data potential"]
        AFl = [r_init, np.log(Jlam_FD[:,0]), "by formula potential"]
        ADl = [r_init, np.log(Jlam_FD[:,1]), "by data potential"]
        # plot_scatter2d_compare_formula_and_data(PFl, PDl, is_show=1)
        # plot_scatter2d_compare_formula_and_data(AFl, ADl, is_show=1)

        # xsp = abs(x[:,0:3])
        # xsp = np.log(x/Js)[:,3:6]
        xsp = np.hstack(( np.array([ r_init ]).T, np.array([ V_init ]).T, np.array([ dlt_Phi ]).T, np.array([ dlt_Jlam ]).T, 
            np.array([ orbitStackeltype_FD[:,0] ]).T, np.array([ orbitStackeltype_FD[:,1] ]).T, 
            np.array([ x[:,0] ]).T, np.array([ x[:,1] ]).T, np.array([ x[:,2] ]).T ))
        xad = np.hstack(( np.array([ r_init ]).T, np.array([ r_init ]).T, np.array([ r_init ]).T ))
        # plot_scatter2d_rate(xsp[:,[0,2]], k_median=0., is_show=1)
        # plot_scatter2d_rate(xsp[:,[0,3]], k_median=0., is_show=1)
        # plot_scatter2d_rate(xsp[:,[0,3]], k_median=0., is_show=1)
        # plot_scatter2d_rate(xsp[:,[2,3]], k_median=1., is_show=1)
        # plot_scatter3d_dd(xsp[:,[0,1,3]], k_median=1., is_show=1)
        # plot_scatter3d_dd(xsp[:,[0,3,4]], k_median=0., is_show=1)
        # plot_scatter3d_dd(xsp[:,[6,7,8]], k_median=0., is_show=1)
        # tx[i] = x

        swit = 0
        r_init = r_init
        TA_F = dos[:, 30:33]
        TA_D = dos[:, -15:-12]
        TA_pp = x_pp
        # add.DEBUG_PRINT_V(0, x_pp, len(x_pp))
        pointsize = 0.2
        fontsize = 6.0
        fig = plt.figure(dpi=300)
        plt.scatter(r_init, abs( TA_D[cl,swit]/TA_F[cl,swit] ), s=pointsize, label="log(D/F)")
        plt.scatter(r_init, abs( TA_pp[cl,swit]/TA_F[cl,swit] ), s=pointsize, label="log(pp/F)")
        # plt.scatter(r_init, abs( TA_pp[cl,swit]/TA_D[cl,swit] ), s=pointsize, label="log(D/F)")
        plt.plot([min(r_init),max(r_init)], [1.,1.], label=r"line $rate=1.$", color="k", lw=pointsize)
        plt.legend()
        plt.yscale("log")
        plt.show()
        plt.close()
