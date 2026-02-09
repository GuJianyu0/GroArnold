#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import analysis_data_distribution as ads
import galaxy_models as gm
import fit_rho_fJ as fff



# [] path
galaxy_name = sys.argv[1]
# galaxy_name = "galaxy_general"
# galaxy_name = "galaxy_general_NFW_spinH_axisLH1"
# galaxy_name = "galaxy_general_Ein_spinH_axisLH0"
# galaxy_name = "galaxy_general_Ein_spinL_axisLH1"
# galaxy_name = "galaxy_general_Ein_spinH_axisLH0_rotvelpot"
# galaxy_name = "galaxy_general_Ein_spinH_axisLH1_rotvelpot"
# galaxy_name = "galaxy_general_Ein_spinH_axisLH2_spininter1_rotvelpot"
# galaxy_name = "galaxy_general_DPLNFW_axisratioz_unmodify0"
# snapshot_ID = 10
snapshot_ID = int(sys.argv[2])
# snapshot_list = [snapshot_ID-1, snapshot_ID]
snapshot_list = [snapshot_ID-2, snapshot_ID-1, snapshot_ID, snapshot_ID+1, snapshot_ID+2]
TimeBetSnapshot = 0.1
time_list = np.array(snapshot_list).astype(float)*TimeBetSnapshot + 0.0
# is_show = True
is_show = False

galaxy_general_location_path = "../../../GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/"
triaxialize_data_path = galaxy_general_location_path+galaxy_name+"/aa/snapshot_%d_triaxialize.txt"
potential_compare_path = galaxy_general_location_path+galaxy_name+"/intermediate/potential_compare_%d_%d.txt"
elliporbit_data_path = galaxy_general_location_path+galaxy_name+"/intermediate/orbit_%d/"
foci_data_path = galaxy_general_location_path+galaxy_name+"/intermediate/snapshot_%d_lmn_foci_Pot.txt"
xv_beforepreprocess_path = galaxy_general_location_path+galaxy_name+"/txt/snapshot_%03d.txt"
aa_data_path = galaxy_general_location_path+galaxy_name+"/aa/snapshot_%d.action.method_all.txt"
save_total_path = galaxy_general_location_path+"/params_statistics/"
save_single_path = galaxy_general_location_path+galaxy_name+"/fit/"



def plot_tau_ptau(filename_folder, ID, savename=""):
    swit_name = [r"\lambda", r"\mu", r"\nu"]

    fontsize = 20.0
    pointsize = 4.
    # figsize = None
    figsize = (10, 10)
    dpi = 200
    colors = ["red", "orange", "yellow", "green", "blue"]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    projection = None

    for swit in np.arange(3):
        ax = fig.add_subplot(3, 1, swit+1)
        if swit==0:
            ax.set_title("particle_ID %d"%(ID))

        for snapshot_ID in snapshot_list:
            filename1 = filename_folder+"tau_ptau_snapshot_%d_particle_%d_swit_%d.txt"%(snapshot_ID, ID, swit)
            data1 = np.loadtxt(filename1, dtype=float)

            index_sort = np.argsort(data1[:,6]) #tau
            data1 = data1[index_sort]
            x1 = data1[:,0:3]
            v1 = data1[:,3:6]
            tau = data1[:,6]
            phichi = data1[:,7]
            Ints = data1[:,8:11]
            # for lambda, ints[0]=E, ints[1]=At[0], ints[2]=Bt[0];
            # for mu, ints[0]=E, ints[1]=At[1], ints[2]=Bt[1];
            # for nu, ints[0]=E, ints[1]=At[2], ints[2]=Bt[2];
            ptau_square1_no_cos = data1[:,11]
            ptau_return1 = data1[:,12]
            tau0_lmn = data1[:,13:16]

            E = Ints[:, 0]
            Atau = Ints[:, 1]
            Btau = Ints[:, 2]
            ptau = ptau_return1
            # ptau = ptau_square1_no_cos**0.5
            # yplot = ptau_square1_no_cos**0.5*np.sign(ptau_square1_no_cos)
            
            label0 = r"$\tau_{0}$ of snapshot %d"%(snapshot_ID) if swit==2 else None
            label = r"$p_{\tau}$ of snapshot %d"%(snapshot_ID) if swit==2 else None
            # label_E    = r"$E$ of snapshot %d"%(snapshot_ID) if swit==2 else None
            # label_Atau = r"$A_{\tau}$ of snapshot %d"%(snapshot_ID) if swit==2 else None
            # label_Btau = r"$B_{\tau}$ of snapshot %d"%(snapshot_ID) if swit==2 else None
            color = colors[snapshot_ID-snapshot_list[0]]

            plt.plot(tau, ptau, lw=pointsize, color=color, label=label)
            plt.axvline(tau0_lmn[0, swit], linestyle='--', color=color, label=label0)
            if swit==2:
                plt.legend(fontsize=fontsize/2, ncol=1, loc="upper right")

        # plt.xscale("log")
        # plt.yscale("log")
        plt.xlabel(r"$%s$"%(swit_name[swit]), fontsize=fontsize)
        plt.ylabel(r"$p_%s$"%(swit_name[swit]), fontsize=fontsize)
    
    fig.tight_layout()
    savename_fig = savename + "_%d"%(ID) + ".pdf"
    fig.tight_layout()
    fig.savefig(
        savename_fig, 
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=dpi,
    )
    if is_show:
        plt.show()
    plt.close(fig)
    print(f"Plot {savename_fig}, done.")

def plot_tau_AtauBtau(filename_folder, ID, savename=""):
    swit_name = [r"\lambda", r"\mu", r"\nu"]
    fudgevalue_name = [r"E", r"A", r"B", r"\chi"]
    n_fudgevalue = len(fudgevalue_name)

    fontsize = 20.0
    pointsize = 4.
    figsize = None
    figsize = (16, 10)
    dpi = 200
    colors = ["red", "orange", "yellow", "green", "blue"]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    projection = None

    for i_ints in np.arange(4):
        for swit in np.arange(3):
            ax = fig.add_subplot(3, n_fudgevalue, i_ints+swit*n_fudgevalue+1)
            if swit==0:
                ax.set_title("particle_ID %d"%(ID))

            for snapshot_ID in snapshot_list:
                filename1 = filename_folder+"tau_ptau_snapshot_%d_particle_%d_swit_%d.txt"%(snapshot_ID, ID, swit)
                data1 = np.loadtxt(filename1, dtype=float)

                index_sort = np.argsort(data1[:,6]) #tau
                data1 = data1[index_sort]
                x1 = data1[:,0:3]
                v1 = data1[:,3:6]
                tau = data1[:,6]
                chiphi = data1[:,7]
                Ints = data1[:,8:11]
                # tau_i (3) is lambda, mu0, nu0
                # for lambda, ints[0]=E, ints[1]=At[0], ints[2]=Bt[0];
                # for mu, ints[0]=E, ints[1]=At[1], ints[2]=Bt[1];
                # for nu, ints[0]=E, ints[1]=At[2], ints[2]=Bt[2];
                ptau_return1 = data1[:,12]
                tau0_lmn = data1[:,13:16]

                E = Ints[:, 0]
                Atau = Ints[:, 1]
                Btau = Ints[:, 2]
                chitau = chiphi
                ptau = ptau_return1

                # E = np.log10(np.abs(E))*np.sign(E)
                # Atau = np.log10(np.abs(Atau))*np.sign(Atau)
                # Btau = np.log10(np.abs(Btau))*np.sign(Btau)
                # E = ads.activate_range(E)
                # Atau = ads.activate_range(Atau)
                # Btau = ads.activate_range(Btau)
                
                label0 = r"$\tau_{0}$ of snapshot %d"%(snapshot_ID) if swit==2 else None
                # label = r"$p_{\tau}$ of snapshot %d"%(snapshot_ID) if swit==2 else None
                label_E    = r"$E$ of snapshot %d"%(snapshot_ID) if swit==2 else None
                label_Atau = r"$A_{\tau}$ of snapshot %d"%(snapshot_ID) if swit==2 else None
                label_Btau = r"$B_{\tau}$ of snapshot %d"%(snapshot_ID) if swit==2 else None
                label_chitau = r"$\chi_{\tau}$ of snapshot %d"%(snapshot_ID) if swit==2 else None
                color = colors[snapshot_ID-snapshot_list[0]]

                if i_ints==0:
                    plt.plot(tau, E, lw=pointsize, color=color, label=label_E)
                if i_ints==1:
                    plt.plot(tau, Atau, lw=pointsize, color=color, label=label_Atau)
                if i_ints==2:
                    plt.plot(tau, Btau, lw=pointsize, color=color, label=label_Btau)
                if i_ints==3:
                    plt.plot(tau, chitau, lw=pointsize, color=color, label=label_chitau)
                plt.axvline(tau0_lmn[0, swit], linestyle='--', color=color, label=label0)
                if swit==2:
                    plt.legend(fontsize=fontsize/3, ncol=1, loc="upper right")

            # plt.xscale("log")
            # plt.yscale("log")
            plt.xlabel(r"$%s$"%(swit_name[swit]), fontsize=fontsize)
            if i_ints==0:
                plt.ylabel(r"$%s$"%(fudgevalue_name[i_ints]), fontsize=fontsize)
            else:
                plt.ylabel(r"$%s_%s$"%(fudgevalue_name[i_ints], swit_name[swit]), fontsize=fontsize)
        
    fig.tight_layout()
    savename_fig = savename + "_%d"%(ID) + ".pdf"
    fig.tight_layout()
    fig.savefig(
        savename_fig, 
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=dpi,
    )
    if is_show:
        plt.show()
    plt.close(fig)
    print(f"Plot {savename_fig}, done.")



if __name__ == '__main__':

    filename_folder = galaxy_general_location_path+galaxy_name+"/debug/"
    savename = save_single_path+"tau_ptau"
    savename2 = save_single_path+"tau_AtauBtau"
    # particle_ID_list = [17]
    # particle_ID_list = [16, 17, 18, 19, 20]
    particle_ID_list = [6015, 6016, 6017, 6018, 6019]
    for ID in particle_ID_list:
        plot_tau_ptau(filename_folder, ID, savename=savename)
        plot_tau_AtauBtau(filename_folder, ID, savename=savename2)
    