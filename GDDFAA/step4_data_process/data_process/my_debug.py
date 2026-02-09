#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import analysis_data_distribution as ads
import galaxy_models as gm
import modify_initial_condition as mic
import plot_action_figs as paf
import triaxialize_galaxy as tg



if __name__ == '__main__':

    data_path_plot = paf.aa_data_path%(paf.snapshot_ID)
    data = np.loadtxt(data_path_plot, dtype=float)
    # N_data = len(data)
    # mass = data[:, 7]
    # x = data[:, 0:0+paf.Dim] #xv
    # v = data[:, paf.Dim:paf.Dim+paf.Dim]
    # X = ads.norm_l(x, axis=1)
    # V = ads.norm_l(v, axis=1)
    # xmed = np.median(X)
    # vmed = np.median(V)
    # xmean = np.mean(X)
    # vmean = np.mean(V)
    # P_F = data[:, 10] #potential
    P_D = data[:, 11]



    # data_path_plot = "/home/darkgaia/workroom/0prog/"\
    #     +"GroArnold_framework/GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/"\
    #     +"galaxy_general_NFW_spinL_axisLH0/aa/snapshot_10.action.method_all.txt"
    # data = np.loadtxt(data_path_plot, dtype=float)
    # pot_old = data[:, 11] #new

    data_path_plot = "/home/darkgaia/workroom/0prog/"\
        +"GroArnold_framework/GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/"\
        +"galaxy_general_Ein_spinH_axisLH0/txt/snapshot_010.txt" #000, 001 high, 002 low (high if no move xv), 003 low, 010 high
        # +"galaxy_general_Ein_spinL_axisLH0/txt/snapshot_010.txt" #004 low, 005, 010 high half
        # +"galaxy_general_NFW_spinL_axisLH0/txt/snapshot_010.txt" #010 low
        # +"galaxy_general_NFW_spinH_axisLH0/txt/snapshot_010.txt" #010 high
        # +"galaxy_general_DPLNFW_axisratioz_unmodify0/txt/snapshot_010.txt"
    data = np.loadtxt(data_path_plot, dtype=float)
    N_data = len(data)
    mass = data[:, 8]
    x = data[:, 0:0+paf.Dim] #xv
    v = data[:, paf.Dim:paf.Dim+paf.Dim]
    X = ads.norm_l(x, axis=1)
    V = ads.norm_l(v, axis=1)
    pot_old = data[:, 10] #potential gadget before tri



    print("xvcenter", np.mean(x, axis=0), np.mean(v, axis=0))
    boundary_rotate_refer = None #kpc, km/s
    # boundary_rotate_refer = [60., 1000.] #kpc, km/s
    # boundary_rotate_refer = [0.1, 1000.] #kpc, km/s
    x, v, operators = tg.all_triaxial_process( #rotate frame
        x, v, mass, is_centralize_coordinate=True, 
        # x, v, mass, is_centralize_coordinate=False, 
        # is_rotate_mainAxisDirection=True, r_range=boundary_rotate_refer, 
        is_rotate_mainAxisDirection=False, r_range=boundary_rotate_refer, 
        # is_eliminate_totalRotation=True, is_by_DF=False, DF=None
        is_eliminate_totalRotation=False, is_by_DF=False, DF=None
    )
    # print("xvcenter", np.mean(x, axis=0), np.mean(v, axis=0))
    # ads.DEBUG_PRINT_V(0, data[0], mass[0])

    spin1_direct = tg.spin_lambda_Nbody(x, v, mass=mass, pot=pot_old)
    fracubd1 = tg.frac_unbounded_particles(v, pot_old)
    ads.DEBUG_PRINT_V(1, v[0], spin1_direct, fracubd1, "\n\n\n\nv[0], lambda1, fracubd1")



    # # x, v = mic.modify_IC(x, v, N_iter=2, is_center_v=True) #1 #2 #6 #np.inf
    # x, v = mic.modify_IC(x, v, N_iter=6, is_center_v=True) #1 #2 #6 #np.inf

    # spin1_direct = tg.spin_lambda_Nbody(x, v, mass=mass, pot=pot_old)
    # fracubd1 = tg.frac_unbounded_particles(v, pot_old)
    # ads.DEBUG_PRINT_V(1, v[0], spin1_direct, fracubd1, "\n\n\n\nv[0], lambda1, fracubd1")



    ## problems:
    #?? rerun and refit, wrong bound in total I and L #?? wrong pot 
    #?? rotate xv decrease spin, do not use rotate v when calculate spin (there models: min 0.003, max 0.03)
    #?? expected snapshot_010 spinLH inverse sometimes (spin varing with time: L 0.003->0.02; H 0.03->0.02, 0.007), 
    #\ xv center move, plot PEL expecially L not conserve, preprocess may influence Ltot much
    
    boundary_rotate_refer = None #kpc, km/s
    # boundary_rotate_refer = [60., 1000.] #kpc, km/s
    # boundary_rotate_refer = [0., 1000.] #kpc, km/s
    # boundary_rotate_refer = [1000., 1000.] #kpc, km/s
    # x, v, operators = tg.all_triaxial_process( #rotate frame
    #     x, v, mass, is_centralize_coordinate=True, 
    #     # x, v, mass, is_centralize_coordinate=False, 
    #     # is_rotate_mainAxisDirection=True, r_range=boundary_rotate_refer, 
    #     is_rotate_mainAxisDirection=False, r_range=boundary_rotate_refer, 
    #     # is_eliminate_totalRotation=True, is_by_DF=False, DF=None
    #     is_eliminate_totalRotation=False, is_by_DF=False, DF=None
    # )
    x, v, operators = tg.all_triaxial_process( #rotate frame
        x, v, mass, is_centralize_coordinate=True, 
        # x, v, mass, is_centralize_coordinate=False, 
        is_rotate_mainAxisDirection=True, r_range=boundary_rotate_refer, 
        # is_rotate_mainAxisDirection=False, boundary_rotate_refer=boundary_rotate_refer, 
        is_eliminate_totalRotation=True, is_by_DF=False, DF=None
        # is_eliminate_totalRotation=False, is_by_DF=False, DF=None
    )
    print("operators: ", operators)

    spin1_direct = tg.spin_lambda_Nbody(x, v, mass=mass, pot=pot_old)
    fracubd1 = tg.frac_unbounded_particles(v, pot_old)
    ads.DEBUG_PRINT_V(1, v[0], spin1_direct, fracubd1, "\n\n\n\nv[0], lambda1, fracubd1")
