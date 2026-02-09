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

def plot_that(tp, t0=0., kt=0.01, is_show=1):

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

if __name__ == "__main__":

    MG = None
    gmn = ""
    # gmn = "_1_NFW"
    # gmn = "_11_NFW_triaxial"
    # gmn = "_41_EinastoUsual"
    galaxymodel_name = "galaxy_general"+gmn+"/"
    RD = fff.Read_data_galaxy(MG, 2, gmn=galaxymodel_name)
    
    snapshot_begin  = 0
    snapshot_end    = 300
    N_snapshot      = 30
    kt = 0.01*(snapshot_end-snapshot_begin)/N_snapshot
    t0 = 0.01*snapshot_begin
    snapshot_sequence = np.linspace(snapshot_begin, snapshot_end, N_snapshot+1)[0:-1].astype(int)
    tx = list(range(N_snapshot))
    for i in np.arange(len(snapshot_sequence)): #snapshot_sequence[i]
        dos0 = RD.data_original_simulationoOutput(i)
        x = dos0[:,0:3]
        tx[i] = x

    rs = 19.6
    n = 1000
    circle_rigid = [rs, [0,0,0], [0,0,0]]
    p = rs, n
    tp = potential_compare_rotate(circle_rigid, tx, *p)
    plot_that(tp, t0=t0, kt=kt, is_show=1)