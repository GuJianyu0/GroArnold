#!/usr/bin/env python
# -*- coding:utf-8 -*-

from gettext import translation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import pdb
from tqdm import tqdm

import galaxy_models as gm
import analysis_data_distribution as add
import fit_rho_fJ as fff

def generate_sample_square(n1=10,n2=10,ls=10.):
    x1 = np.arange(n1)*ls*6/n1
    x1 -= np.max(x1)/2
    x2 = np.arange(n2)*ls*6/n2
    x2 -= np.max(x2)/2
    X, Y = np.meshgrid(x1, x2)
    Z = ls-X-Y
    return X,Y,Z

if __name__ == "__main__":

    ##data
    MG = None
    gm_name = ""
    # gm_name = "_1_NFW_spherical"
    # gm_name = "_4_EinastoUsual_spherical"
    # gm_name = "_11_NFW_triaxial"
    # gm_name = "_41_EinastoUsual_triaxial"
    galaxymodel_name = "galaxy_general"+gm_name+"/"
    snapshot_Id = 000
    # snapshot_Id = 500

    whatcannonical = 2
    method_and_tags = "C2P0S0A0" #density
    # method_and_tags = "C5P0S0A0" #SphFP
    # method_and_tags = "C5P0S2A0" #SFFP
    # method_and_tags = "C5P1S2A0" #SFDP
    # method_and_tags = "C5P1S2A1" #PPOD
    # method_and_tags = "C5P1S2A2" #Comb-lmn
    bd = np.inf
    bd_display = bd

    galaxymodel_name = "galaxy_general"+gm_name+"/"
    RD = fff.Read_data_galaxy(MG, gmn=galaxymodel_name, wc=whatcannonical) #No such type of data provided
    # doc = RD.data_secondhand_snapshot(snapshot_Id, wc=whatcannonical)
    doc = RD.data_original_NDFA(snapshot_Id, method_and_tags=method_and_tags)
    x, y = RD.data_sample_screen(doc, x_down=1./bd, x_up=bd, is_logy=True)
    xerr = x*0.
    yerr = y*0.1
    x = x #now x[3], v[3]
    res = gm.position_estimateScale(x)
    print("AA scale from data: %e" % res)

    ds = 0.00456
    ls = 19.6
    qy = 0.6
    qz = 0.3

    ##pot
    n1 = 30
    n2 = 30
    X,Y,Z = generate_sample_square(n1,n2,ls)

    PD = np.zeros((n1,n2))
    PF = np.zeros((n1,n2))
    for i in np.arange(n1):
        for j in np.arange(n2):
            target = np.array([X[i,j], Y[i,j], Z[i,j]])
            PD[i,j] = gm.potential_nbody_simple(target, x)
            PF[i,j] = gm.Phi_doublepowerlaw_NFW_triaxial(target, ds,ls,qy,qz)
    PD = np.log(-PD)/np.log(10.)
    PF = np.log(-PF)/np.log(10.)
    # add.DEBUG_PRINT_V(1, X,Y,Z, PD,PF,"grid")
    ##many sanpshots??

    ##plot
    pointsize = 0.2
    fontsize = 6.0
    dpi = 500
    fig = plt.figure(dpi=dpi)
    # fig = plt.figure(dpi=300, figsize=(10,2)) #show, not axis length
    ax=fig.add_subplot(1,1,1, projection='3d')
    ax.grid(True)

    # cm = plt.cm.get_cmap('gist_rainbow') #rainbow
    # axsc = ax.scatter(Xl[:,0], Yl[:,1], Zl[:,2], s=pointsize, label="", c=Pl, cmap=cm)
    # plt.colorbar(axsc)

    surf = ax.plot_surface(X, Y, PD, cmap=cm.jet, linewidth=0, antialiased=False, label="potential by nbody data")
    surf = ax.plot_surface(X, Y, PF, cmap=cm.jet, linewidth=0, antialiased=False, label="potential by fit formula")
    position=fig.add_axes([0.1, 0.3, 0.02, 0.5]) #x pos, ypos, width, shrink
    fig.colorbar(surf, cax=position, aspect=5)

    ax.set_xlabel(r"x", fontsize=fontsize)
    ax.set_ylabel(r"y", fontsize=fontsize)
    ax.set_zlabel(r"z", fontsize=fontsize)
    ax.set_zlabel(r"potential", fontsize=fontsize)
    # ax.set_title(r"potential", fontsize=fontsize)
    # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)
    plt.show()